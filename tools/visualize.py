import argparse
import copy
import os
from typing import Optional, Tuple

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def _filter_by_classes(
    bboxes: np.ndarray,
    labels: np.ndarray,
    bbox_classes: Optional[list],
    scores: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if bbox_classes is None:
        return bboxes, labels, scores

    indices = np.isin(labels, bbox_classes)
    bboxes = bboxes[indices]
    labels = labels[indices]
    if scores is not None:
        scores = scores[indices]

    return bboxes, labels, scores


def _filter_by_score(
    bboxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray],
    bbox_score: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if bbox_score is None or scores is None:
        return bboxes, labels, scores

    indices = scores >= bbox_score
    bboxes = bboxes[indices]
    labels = labels[indices]
    scores = scores[indices]

    return bboxes, labels, scores


def _to_lidar_boxes(bboxes: np.ndarray) -> LiDARInstance3DBoxes:
    # Original visualize.py expected bottom-centered boxes.
    bboxes = bboxes.copy()
    bboxes[..., 2] -= bboxes[..., 5] / 2
    return LiDARInstance3DBoxes(bboxes, box_dim=9)


def build_gt_boxes(data, args):
    if "gt_bboxes_3d" not in data:
        return None, None

    bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
    labels = data["gt_labels_3d"].data[0][0].numpy()

    bboxes, labels, _ = _filter_by_classes(
        bboxes=bboxes,
        labels=labels,
        bbox_classes=args.bbox_classes,
    )

    if len(bboxes) == 0:
        return None, None

    return _to_lidar_boxes(bboxes), labels


def build_pred_boxes(outputs, idx, args):
    if outputs is None or "boxes_3d" not in outputs[idx]:
        return None, None

    bboxes = outputs[idx]["boxes_3d"].tensor.numpy()
    scores = outputs[idx]["scores_3d"].numpy()
    labels = outputs[idx]["labels_3d"].numpy()

    bboxes, labels, scores = _filter_by_classes(
        bboxes=bboxes,
        labels=labels,
        scores=scores,
        bbox_classes=args.bbox_classes,
    )

    bboxes, labels, scores = _filter_by_score(
        bboxes=bboxes,
        labels=labels,
        scores=scores,
        bbox_score=args.bbox_score,
    )

    if len(bboxes) == 0:
        return None, None

    return _to_lidar_boxes(bboxes), labels


def build_masks(data, outputs, idx, args):
    if args.mode == "gt" and "gt_masks_bev" in data:
        masks = data["gt_masks_bev"].data[0].numpy()
        return masks.astype(np.bool_)

    if args.mode in ["pred", "both"] and outputs is not None and "masks_bev" in outputs[idx]:
        masks = outputs[idx]["masks_bev"].numpy()
        return masks >= args.map_score

    return None


def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["gt", "pred", "both"],
        help="Visualization mode: gt, pred, or both.",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # Build dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    model = None
    outputs = None

    # Build model and run inference only when prediction is required.
    if args.mode in ["pred", "both"]:
        if args.checkpoint is None:
            raise ValueError("--mode pred/both requires --checkpoint")

        cfg.model.pretrained = None
        cfg.model.train_cfg = None

        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            model.CLASSES = dataset.CLASSES

        print(model.CLASSES)

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )

        model.eval()
        outputs = multi_gpu_test(
            model,
            dataflow,
            tmpdir=None,
            gpu_collect=False,
        )

    for idx, data in enumerate(tqdm(dataflow)):
        metas = data["metas"].data[0][0]
        name = "{}-{}".format(metas["timestamp"], metas["token"])

        gt_bboxes, gt_labels = (None, None)
        pred_bboxes, pred_labels = (None, None)

        if args.mode in ["gt", "both"]:
            gt_bboxes, gt_labels = build_gt_boxes(data, args)

        if args.mode in ["pred", "both"]:
            pred_bboxes, pred_labels = build_pred_boxes(outputs, idx, args)

        # If --bbox-classes is specified, save only frames that contain
        # at least one selected class in GT or Prediction.
        # - gt mode   : requires selected GT bbox
        # - pred mode : requires selected Pred bbox
        # - both mode : requires selected GT or Pred bbox
        if args.bbox_classes is not None:
            has_gt = gt_bboxes is not None and len(gt_bboxes) > 0
            has_pred = pred_bboxes is not None and len(pred_bboxes) > 0

            if not (has_gt or has_pred):
                continue

        masks = build_masks(data, outputs, idx, args)

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)

                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    gt_bboxes=gt_bboxes if args.mode in ["gt", "both"] else None,
                    gt_labels=gt_labels if args.mode in ["gt", "both"] else None,
                    pred_bboxes=pred_bboxes if args.mode in ["pred", "both"] else None,
                    pred_labels=pred_labels if args.mode in ["pred", "both"] else None,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                    # GT: green, Prediction: red
                    gt_color=(0, 255, 0),
                    pred_color=(255, 0, 0),
                )

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()

            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                gt_bboxes=gt_bboxes if args.mode in ["gt", "both"] else None,
                gt_labels=gt_labels if args.mode in ["gt", "both"] else None,
                pred_bboxes=pred_bboxes if args.mode in ["pred", "both"] else None,
                pred_labels=pred_labels if args.mode in ["pred", "both"] else None,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
                # GT: green, Prediction: red
                gt_color=(0, 255, 0),
                pred_color=(255, 0, 0),
            )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()