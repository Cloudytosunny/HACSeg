#!/usr/bin/env python3
"""
unified_evaluate_ins_decoupled_pq.py - IPQth Evaluation Script Supporting Both SemanticKITTI and nuScenes

Features:
1. Select evaluation targets—either SemanticKITTI (.label) or nuScenes (panoptic .npz) predictions—via the `--dataset` argument.
2. Automatically construct INS-GT, perform in-memory instance-to-semantic alignment, and compute PQ, SQ, RQ, and IoU metrics.
3. The output format aligns with the official nuScenes IPQ evaluation standard; it uniformly displays statistics for "Things" classes and supports saving reports in YAML or HTML format.

使用示例：
    # SemanticKITTI
    python unified_evaluate_ins_decoupled_pq.py \
        --dataset semantickitti \
        --pred-dir /path/to/predictions \
        --gt-dir /path/to/labels \
        --output ./results_sem


    # nuScenes
    python unified_evaluate_ins_decoupled_pq.py \
        --dataset nuscenes \
        --result-path /path/to/panoptic_results \
        --dataroot /data/sets/nuscenes \
        --version v1.0-trainval \
        --eval-set val \
        --output ./results_nus

"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import yaml
from tqdm import tqdm

try:
    from nuscenes.eval.panoptic.panoptic_seg_evaluator import PanopticEval  # type: ignore
except Exception:  # pragma: no cover - fallback for纯SK使用场景
    class PanopticEval:  # noqa: D401 - fallback实现与SemanticKITTI相同
        """Panoptic evaluation using numpy."""

        def __init__(self, n_classes, ignore=None, offset=2 ** 32, min_points=30):
            self.n_classes = n_classes
            self.ignore = np.array(ignore, dtype=np.int64)
            self.include = np.array([n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
            self.reset()
            self.offset = offset
            self.min_points = min_points
            self.eps = 1e-15

        def reset(self):
            self.px_iou_conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
            self.pan_tp = np.zeros(self.n_classes, dtype=np.int64)
            self.pan_iou = np.zeros(self.n_classes, dtype=np.double)
            self.pan_fp = np.zeros(self.n_classes, dtype=np.int64)
            self.pan_fn = np.zeros(self.n_classes, dtype=np.int64)

        def addBatchSemIoU(self, x_sem, y_sem):
            idxs = np.stack([x_sem, y_sem], axis=0)
            np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

        def getSemIoU(self):
            conf = self.px_iou_conf_matrix.copy().astype(np.double)
            conf[:, self.ignore] = 0
            tp = conf.diagonal()
            fp = conf.sum(axis=1) - tp
            fn = conf.sum(axis=0) - tp
            intersection = tp
            union = tp + fp + fn
            union = np.maximum(union, self.eps)
            iou = intersection.astype(np.double) / union.astype(np.double)
            iou_mean = (intersection[self.include].astype(np.double) / union[self.include].astype(np.double)).mean()
            return iou_mean, iou

        def addBatchPanoptic(self, x_sem_row, x_inst_row, y_sem_row, y_inst_row):
            x_inst_row = x_inst_row + 1
            y_inst_row = y_inst_row + 1

            for cl in self.ignore:
                mask = y_sem_row != cl
                x_sem_row = x_sem_row[mask]
                y_sem_row = y_sem_row[mask]
                x_inst_row = x_inst_row[mask]
                y_inst_row = y_inst_row[mask]

            for cl in self.include:
                x_mask = x_sem_row == cl
                y_mask = y_sem_row == cl
                x_inst_in_cl = x_inst_row * x_mask.astype(np.int64)
                y_inst_in_cl = y_inst_row * y_mask.astype(np.int64)

                unique_pred, counts_pred = np.unique(x_inst_in_cl[x_inst_in_cl > 0], return_counts=True)
                id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
                matched_pred = np.array([False] * unique_pred.shape[0])

                unique_gt, counts_gt = np.unique(y_inst_in_cl[y_inst_in_cl > 0], return_counts=True)
                id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
                matched_gt = np.array([False] * unique_gt.shape[0])

                valid = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
                offset_combo = x_inst_in_cl[valid] + self.offset * y_inst_in_cl[valid]
                unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

                if unique_combo.size > 0:
                    gt_labels = unique_combo // self.offset
                    pred_labels = unique_combo % self.offset
                    gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
                    pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
                    intersections = counts_combo
                    unions = gt_areas + pred_areas - intersections
                    ious = intersections.astype(np.double) / unions.astype(np.double)

                    tp_idx = ious > 0.5
                    self.pan_tp[cl] += np.sum(tp_idx)
                    self.pan_iou[cl] += np.sum(ious[tp_idx])
                    matched_gt[[id2idx_gt[id] for id in gt_labels[tp_idx]]] = True
                    matched_pred[[id2idx_pred[id] for id in pred_labels[tp_idx]]] = True

                self.pan_fn[cl] += np.sum(np.logical_and(counts_gt >= self.min_points, matched_gt == False))
                self.pan_fp[cl] += np.sum(np.logical_and(counts_pred >= self.min_points, matched_pred == False))

        def getPQ(self):
            sq_all = self.pan_iou.astype(np.double) / np.maximum(self.pan_tp.astype(np.double), self.eps)
            rq_all = self.pan_tp.astype(np.double) / np.maximum(
                self.pan_tp.astype(np.double) + 0.5 * self.pan_fp.astype(np.double) + 0.5 * self.pan_fn.astype(np.double),
                self.eps,
            )
            pq_all = sq_all * rq_all
            SQ = sq_all[self.include].mean()
            RQ = rq_all[self.include].mean()
            PQ = pq_all[self.include].mean()
            return PQ, SQ, RQ, pq_all, sq_all, rq_all

        def addBatch(self, x_sem, x_inst, y_sem, y_inst):
            self.addBatchSemIoU(x_sem, y_sem)
            self.addBatchPanoptic(x_sem, x_inst, y_sem, y_inst)


# SemanticKITTI 配置
LEARNING_MAP = {
    0: 0,
    1: 0,
    10: 1,
    11: 2,
    13: 5,
    15: 3,
    16: 5,
    18: 4,
    20: 5,
    30: 6,
    31: 7,
    32: 8,
    40: 9,
    44: 10,
    48: 11,
    49: 12,
    50: 13,
    51: 14,
    52: 0,
    60: 9,
    70: 15,
    71: 16,
    72: 17,
    80: 18,
    81: 19,
    99: 0,
    252: 1,
    253: 7,
    254: 6,
    255: 8,
    256: 5,
    257: 5,
    258: 4,
    259: 5,
}

LEARNING_MAP_INV = {
    0: 0,
    1: 10,
    2: 11,
    3: 15,
    4: 18,
    5: 20,
    6: 30,
    7: 31,
    8: 32,
    9: 40,
    10: 44,
    11: 48,
    12: 49,
    13: 50,
    14: 51,
    15: 70,
    16: 71,
    17: 72,
    18: 80,
    19: 81,
}

CLASS_STRINGS = {
    0: "unlabeled",
    1: "car",
    2: "bicycle",
    3: "motorcycle",
    4: "truck",
    5: "other-vehicle",
    6: "person",
    7: "bicyclist",
    8: "motorcyclist",
    9: "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "terrain",
    18: "pole",
    19: "traffic-sign",
}

SEMANTICKITTI_THINGS = ["car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist"]
THING_LEARNING_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
MAX_LABEL = max(LEARNING_MAP.keys()) + 100
LEARNING_MAP_ARRAY = np.zeros(MAX_LABEL, dtype=np.int32)
for k, v in LEARNING_MAP.items():
    if k < MAX_LABEL:
        LEARNING_MAP_ARRAY[k] = v


@dataclass
class EvaluationResult:
    dataset_name: str
    thing_names: List[str]
    output_dict: Dict[str, Dict[str, float]]
    thing_summary: Dict[str, float]
    missing_entries: List[str]
    mismatched_entries: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified INS-Decoupled PQ Evaluation Script (SemanticKITTI / nuScenes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["semantickitti", "nuscenes"], required=True, help="Specify the dataset for evaluation.")
    # SemanticKITTI
    parser.add_argument("--pred-dir", type=Path, help="SemanticKITTI Predicted Labels Directory (containing *.label files)")
    parser.add_argument("--gt-dir", type=Path, help="SemanticKITTI Ground Truth Label Directory (containing *.label files)")
    # nuScenes
    parser.add_argument("--result-path", type=Path, help="nuScenes Panoptic Prediction Root Directory (panoptic/<split>/...)")
    parser.add_argument("--dataroot", type=Path, help="nuScenes Data Root Directory (samples/sweeps/panoptic)")
    parser.add_argument("--version", type=str, default="v1.0-trainval", help="nuScenes Version")
    parser.add_argument("--eval-set", type=str, default="val", help="nuScenes evaluation splits, such as val/mini_val.")

    parser.add_argument(
        "--min-inst-points",
        type=int,
        default=None,
        help="Minimum Instance Point Threshold (Default: nuScenes: 15 / SemanticKITTI: 50)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Result Output Directory (Saves scores.txt and HTML)")
    parser.add_argument("--verbose", action="store_true", help="Print more debugging information in nuScenes mode.")
    args = parser.parse_args()

    if args.dataset == "semantickitti":
        if args.pred_dir is None or args.gt_dir is None:
            parser.error("The SemanticKITTI mode requires providing both --pred-dir and --gt-dir.")
    else:
        missing = [flag for flag, val in [("--result-path", args.result_path), ("--dataroot", args.dataroot)] if val is None]
        if missing:
            parser.error(f"nuScenes mode is missing parameters: {', '.join(missing)}}")

    if args.min_inst_points is None:
        args.min_inst_points = 50 if args.dataset == "semantickitti" else 15
    return args


def summarize_results(class_metrics: Dict[str, Dict[str, float]], thing_names: List[str]) -> Dict[str, float]:
    if not thing_names:
        return {"pq_things": 0.0, "rq_things": 0.0, "sq_things": 0.0, "iou_things": 0.0}
    pq = [class_metrics[name]["PQ"] for name in thing_names if name in class_metrics]
    rq = [class_metrics[name]["RQ"] for name in thing_names if name in class_metrics]
    sq = [class_metrics[name]["SQ"] for name in thing_names if name in class_metrics]
    iou = [class_metrics[name]["IoU"] for name in thing_names if name in class_metrics]
    def safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0
    return {
        "pq_things": safe_mean(pq),
        "rq_things": safe_mean(rq),
        "sq_things": safe_mean(sq),
        "iou_things": safe_mean(iou),
    }


def construct_ins_gt_semantic(pred_label: np.ndarray, gt_label: np.ndarray) -> np.ndarray:
    pred_sem_raw = pred_label & 0xFFFF
    gt_sem_raw = gt_label & 0xFFFF
    gt_ins = gt_label >> 16
    pred_sem_learn = LEARNING_MAP_ARRAY[np.clip(pred_sem_raw, 0, MAX_LABEL - 1)]
    ins_gt_label = np.zeros_like(gt_label, dtype=np.uint32)
    unique_gt_instances = np.unique(gt_ins[gt_ins > 0])

    for g_id in unique_gt_instances:
        mask_g = gt_ins == g_id
        votes = pred_sem_learn[mask_g]
        thing_votes = votes[np.isin(votes, THING_LEARNING_IDS)]
        if len(thing_votes) == 0:
            continue
        best_sem_learn = np.bincount(thing_votes).argmax()
        orig_sem_id = LEARNING_MAP_INV.get(best_sem_learn, 0)
        intersect = mask_g & (pred_sem_learn == best_sem_learn)
        if intersect.any():
            ins_gt_label[intersect] = (np.uint32(g_id) << 16) | np.uint32(orig_sem_id)
    return ins_gt_label


def evaluate_semantickitti(args: argparse.Namespace) -> EvaluationResult:
    pred_files = sorted(args.pred_dir.glob("*.label"))
    if not pred_files:
        raise FileNotFoundError(f"No *.label prediction files found in {args.pred_dir}.")

    evaluator = PanopticEval(n_classes=20, ignore=[0], min_points=args.min_inst_points)
    missing, mismatched = [], []

    for pred_file in tqdm(pred_files, desc="Evaluating IPQ"):
        gt_file = args.gt_dir / pred_file.name
        if not gt_file.exists():
            missing.append(pred_file.name)
            continue
        pred_label = np.fromfile(pred_file, dtype=np.uint32)
        gt_label = np.fromfile(gt_file, dtype=np.uint32)
        if len(pred_label) != len(gt_label):
            mismatched.append(pred_file.name)
            continue
        ins_gt_label = construct_ins_gt_semantic(pred_label, gt_label)
        ins_gt_sem = LEARNING_MAP_ARRAY[np.clip(ins_gt_label & 0xFFFF, 0, MAX_LABEL - 1)]
        ins_gt_inst = ins_gt_label
        pred_sem = LEARNING_MAP_ARRAY[np.clip(pred_label & 0xFFFF, 0, MAX_LABEL - 1)]
        pred_inst = pred_label
        evaluator.addBatch(pred_sem, pred_inst, ins_gt_sem, ins_gt_inst)

    pq_mean, sq_mean, rq_mean, pq_all, sq_all, rq_all = evaluator.getPQ()
    iou_mean, iou_all = evaluator.getSemIoU()

    pq_all = pq_all.flatten().tolist()
    sq_all = sq_all.flatten().tolist()
    rq_all = rq_all.flatten().tolist()
    iou_all = iou_all.flatten().tolist()

    output_dict: Dict[str, Dict[str, float]] = {
        "all": {
            "PQ": float(pq_mean),
            "SQ": float(sq_mean),
            "RQ": float(rq_mean),
            "IoU": float(iou_mean),
        }
    }
    for idx, (pq, rq, sq, iou) in enumerate(zip(pq_all, rq_all, sq_all, iou_all)):
        class_str = CLASS_STRINGS.get(idx, f"class_{idx}")
        output_dict[class_str] = {"PQ": float(pq), "SQ": float(sq), "RQ": float(rq), "IoU": float(iou)}

    thing_summary = summarize_results(output_dict, SEMANTICKITTI_THINGS)
    return EvaluationResult(
        dataset_name="SemanticKITTI",
        thing_names=SEMANTICKITTI_THINGS,
        output_dict=output_dict,
        thing_summary=thing_summary,
        missing_entries=missing,
        mismatched_entries=mismatched,
    )


def construct_ins_gt_nuscenes(
    pred_sem: np.ndarray,
    gt_sem: np.ndarray,
    gt_inst: np.ndarray,
    thing_ids: Sequence[int],
    ignore_idx: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ins_sem = np.full(gt_sem.shape, ignore_idx, dtype=np.int32)
    ins_inst = np.zeros(gt_inst.shape, dtype=np.int64)
    thing_ids_array = np.array(sorted(set(thing_ids)), dtype=np.int32)
    unique_instances = np.unique(gt_inst)
    unique_instances = unique_instances[unique_instances > 0]

    for inst_value in unique_instances:
        mask = gt_inst == inst_value
        if not np.any(mask):
            continue
        gt_sem_values = gt_sem[mask]
        if gt_sem_values.size == 0:
            continue
        gt_majority = int(np.bincount(gt_sem_values, minlength=num_classes).argmax())
        if gt_majority not in thing_ids_array:
            continue

        votes = pred_sem[mask]
        thing_vote_mask = np.isin(votes, thing_ids_array)
        thing_votes = votes[thing_vote_mask]
        if thing_votes.size == 0:
            continue
        best_sem = int(np.bincount(thing_votes, minlength=num_classes).argmax())
        intersect_mask = mask & (pred_sem == best_sem)
        if not np.any(intersect_mask):
            continue
        ins_sem[intersect_mask] = best_sem
        ins_inst[intersect_mask] = gt_inst[intersect_mask]

    return ins_sem, ins_inst


def evaluate_nuscenes(args: argparse.Namespace) -> EvaluationResult:
    from nuscenes.eval.panoptic.utils import PanopticClassMapper, get_samples_in_panoptic_eval_set  # type: ignore
    from nuscenes.nuscenes import NuScenes  # type: ignore
    from nuscenes.utils.data_io import load_bin_file  # type: ignore

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=args.verbose)
    mapper = PanopticClassMapper(nusc)
    ignore_idx = mapper.ignore_class["index"]
    num_classes = len(mapper.coarse_name_2_coarse_idx_mapping)
    id2name = {idx: name for name, idx in mapper.coarse_name_2_coarse_idx_mapping.items()}
    thing_names = [name for name, _ in sorted(mapper.things.items(), key=lambda x: x[1])]
    thing_ids = [mapper.things[name] for name in thing_names]

    sample_tokens = get_samples_in_panoptic_eval_set(nusc, args.eval_set)
    evaluator = PanopticEval(n_classes=num_classes, ignore=[ignore_idx], min_points=args.min_inst_points)

    missing, mismatched = [], []
    for sample_token in tqdm(sample_tokens, desc="Evaluating IPQ", disable=False):
        sample = nusc.get("sample", sample_token)
        sd_token = sample["data"]["LIDAR_TOP"]
        gt_panoptic_token = nusc.get("panoptic", sd_token)
        gt_file = Path(nusc.dataroot) / gt_panoptic_token["filename"]
        gt_panoptic = load_bin_file(str(gt_file), type="panoptic")
        gt_sem = mapper.convert_label((gt_panoptic // 1000).astype(np.int32)).astype(np.int32)
        gt_inst = gt_panoptic

        pred_file = args.result_path / "panoptic" / args.eval_set / f"{sd_token}_panoptic.npz"
        if not pred_file.exists():
            missing.append(sd_token)
            continue
        pred_panoptic = load_bin_file(str(pred_file), type="panoptic")
        if pred_panoptic.shape != gt_panoptic.shape:
            mismatched.append(sd_token)
            continue
        pred_sem = (pred_panoptic // 1000).astype(np.int32)
        pred_inst = pred_panoptic

        ins_sem, ins_inst = construct_ins_gt_nuscenes(
            pred_sem=pred_sem,
            gt_sem=gt_sem,
            gt_inst=gt_inst,
            thing_ids=thing_ids,
            ignore_idx=ignore_idx,
            num_classes=num_classes,
        )
        evaluator.addBatch(pred_sem, pred_inst, ins_sem, ins_inst)

    pq_mean, sq_mean, rq_mean, pq_all, sq_all, rq_all = evaluator.getPQ()
    iou_mean, iou_all = evaluator.getSemIoU()

    pq_all = pq_all.flatten().tolist()
    sq_all = sq_all.flatten().tolist()
    rq_all = rq_all.flatten().tolist()
    iou_all = iou_all.flatten().tolist()

    output_dict: Dict[str, Dict[str, float]] = {}
    for idx, (pq, rq, sq, iou) in enumerate(zip(pq_all, rq_all, sq_all, iou_all)):
        class_name = id2name.get(idx, f"class_{idx}")
        output_dict[class_name] = {"PQ": float(pq), "SQ": float(sq), "RQ": float(rq), "IoU": float(iou)}
    output_dict["all"] = {"PQ": float(pq_mean), "SQ": float(sq_mean), "RQ": float(rq_mean), "IoU": float(iou_mean)}

    thing_summary = summarize_results(output_dict, thing_names)
    return EvaluationResult(
        dataset_name="nuScenes",
        thing_names=thing_names,
        output_dict=output_dict,
        thing_summary=thing_summary,
        missing_entries=missing,
        mismatched_entries=mismatched,
    )


def print_results(result: EvaluationResult, elapsed: float) -> None:
    print(f"\nCompleted {result.dataset_name} IPQ evaluation in {elapsed:.2f} seconds.")
    print("=" * 80)
    print(f"{result.dataset_name} IPQ Results (Things Category)")
    print("=" * 80)
    print(f"{'Class':<24} | {'PQ':<10} | {'SQ':<10} | {'RQ':<10} | {'IoU':<10}")
    print("-" * 80)
    for class_name in result.thing_names:
        metrics = result.output_dict.get(class_name, {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "IoU": 0.0})
        print(
            f"{class_name:<24} | {metrics['PQ']:.4f}     | {metrics['SQ']:.4f}     | "
            f"{metrics['RQ']:.4f}     | {metrics['IoU']:.4f}"
        )
    print("-" * 80)
    print(
        f"{'Mean (Things)':<24} | {result.thing_summary['pq_things']:.6f}     | "
        f"{result.thing_summary['sq_things']:.6f}     | {result.thing_summary['rq_things']:.6f}     | "
        f"{result.thing_summary['iou_things']:.6f}"
    )
    print("=" * 80)

    if result.missing_entries:
        print(f"\nWarning: {len(result.missing_entries)} samples are missing matches: {result.missing_entries[:5]}")
    if result.mismatched_entries:
        print(f"Warning: {len(result.mismatched_entries)} number of mismatched sample points: {result.mismatched_entries[:5]}")


def save_outputs(output_dir: Path, result: EvaluationResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = result.output_dict.get("all", {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "IoU": 0.0})
    scores = {
        **result.thing_summary,
        "pq_all": float(all_metrics["PQ"]),
        "sq_all": float(all_metrics["SQ"]),
        "rq_all": float(all_metrics["RQ"]),
        "miou_all": float(all_metrics["IoU"]),
    }
    for class_name in result.thing_names:
        metrics = result.output_dict.get(class_name, {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "IoU": 0.0})
        scores[f"pq_{class_name}"] = float(metrics["PQ"])
        scores[f"sq_{class_name}"] = float(metrics["SQ"])
        scores[f"rq_{class_name}"] = float(metrics["RQ"])
        scores[f"iou_{class_name}"] = float(metrics["IoU"])

    scores_file = output_dir / "scores.txt"
    with open(scores_file, "w", encoding="utf-8") as f:
        yaml.dump(scores, f, allow_unicode=True, default_flow_style=False)
    print(f"YAML results written to: {scores_file}")

    table_data = []
    for class_name in result.thing_names:
        metrics = result.output_dict.get(class_name, {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "IoU": 0.0})
        table_data.append(
            {
                "class": class_name,
                "pq": f"{metrics['PQ']:.4f}",
                "sq": f"{metrics['SQ']:.4f}",
                "rq": f"{metrics['RQ']:.4f}",
                "iou": f"{metrics['IoU']:.4f}",
            }
        )

    html_file = output_dir / "detailed_results.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(
            """<!doctype html>
<html lang="en" style="scroll-behavior: smooth;">
<head>
  <meta charset="UTF-8">
  <title>Unified INS-Decoupled PQ Results</title>
  <script src='https://cdnjs.cloudflare.com/ajax/libs/tabulator/4.4.3/js/tabulator.min.js'></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tabulator/4.4.3/css/bulma/tabulator_bulma.min.css">
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    h1 { color: #333; }
    .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    .summary h2 { margin-top: 0; }
  </style>
</head>
<body>
  <h1>Unified INS-Decoupled PQ Results (Things)</h1>
  <div class="summary">
    <h2>Summary</h2>
    <p><strong>PQ Things:</strong> """
            + f"{result.thing_summary['pq_things']:.10f}"
            + """</p>
    <p><strong>SQ Things:</strong> """
            + f"{result.thing_summary['sq_things']:.10f}"
            + """</p>
    <p><strong>RQ Things:</strong> """
            + f"{result.thing_summary['rq_things']:.10f}"
            + """</p>
    <p><strong>IoU Things:</strong> """
            + f"{result.thing_summary['iou_things']:.10f}"
            + """</p>
  </div>
  <h2>Per-Class Results</h2>
  <div id="classwise_results"></div>
<script>
  let table_data = """
            + json.dumps(table_data)
            + """
  new Tabulator("#classwise_results", {
    layout: "fitColumns",
    data: table_data,
    columns: [
      {title: "Class", field:"class", width:200},
      {title: "PQ", field:"pq", width:110, align: "center"},
      {title: "SQ", field:"sq", width:110, align: "center"},
      {title: "RQ", field:"rq", width:110, align: "center"},
      {title: "IoU", field:"iou", width:110, align: "center"}
    ]
  });
</script>
</body>
</html>"""
        )
    print(f"HTML results have been written to: {html_file}")


def main():
    args = parse_args()
    start_time = time.time()

    if args.dataset == "semantickitti":
        print("*" * 80)
        print("SemanticKITTI INS-Decoupled PQ Evaluation")
        print("*" * 80)
        print(f"Prediction Directory: {args.pred_dir}")
        print(f"Ground Truth Directory: {args.gt_dir}")
        print(f"Minimum Instance Points: {args.min_inst_points}")
        print(f"Output Directory: {args.output}")
        print("*" * 80)
        result = evaluate_semantickitti(args)
    else:
        print("*" * 80)
        print("nuScenes INS-Decoupled PQ Evaluation")
        print("*" * 80)
        print(f"Prediction Directory: {args.result_path}")
        print(f"Data Root Directory: {args.dataroot}")
        print(f"Version: {args.version}")
        print(f"Evaluation split: {args.eval_set}")
        print(f"Minimum Instance Points: {args.min_inst_points}")
        print(f"Output Directory: {args.output}")
        print("*" * 80)
        result = evaluate_nuscenes(args)

    elapsed = time.time() - start_time
    print_results(result, elapsed)
    if args.output is not None:
        save_outputs(args.output, result)


if __name__ == "__main__":
    main()
