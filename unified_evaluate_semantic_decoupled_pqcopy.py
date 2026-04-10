#!/usr/bin/env python3
"""
unified_evaluate_semantic_decoupled_pq.py — An S_PQ evaluation script supporting both SemanticKITTI and nuScenes.

Features:
1. Selects the S_PQ evaluation target—either SemanticKITTI (.label) or nuScenes (panoptic .npz)—via the `--dataset` argument.
2. Constructs CSI-GT (by copying GT instance IDs) to evaluate PQ_csi; subsequently evaluates the original predictions (PQ_pre) and computes S_PQ = PQ_pre / PQ_csi.
3. The output format remains consistent with the nuScenes `nuscenes_Spq.py` script, supporting metrics for Things, Stuff, and All categories, as well as PQ_dagger, and generating reports in YAML and HTML formats.

Example:
    # SemanticKITTI
    python unified_evaluate_semantic_decoupled_pq.py \
        --dataset semantickitti \
        --pred-dir /path/to/predictions \
        --gt-dir /path/to/labels \
        --output ./spq_sem


    # nuScenes
    python unified_evaluate_semantic_decoupled_pq.py \
        --dataset nuscenes \
        --result-path /path/to/panoptic_results \
        --dataroot /data/sets/nuscenes \
        --version v1.0-trainval \
        --eval-set val \
        --output ./spq_nus

"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import yaml
from tqdm import tqdm

from nuscenes.eval.panoptic.panoptic_seg_evaluator import PanopticEval  # type: ignore

# SemanticKITTI 常量
SK_LEARNING_MAP = {
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
SK_CLASS_STRINGS = {
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
SK_THINGS = ["car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist"]
SK_STUFF = [
    "road",
    "sidewalk",
    "parking",
    "other-ground",
    "building",
    "vegetation",
    "trunk",
    "terrain",
    "fence",
    "pole",
    "traffic-sign",
]
SK_ALL_CLASSES = SK_THINGS + SK_STUFF
SK_TARGET_SEMANTIC_IDS = {
    10,
    11,
    13,
    15,
    16,
    18,
    20,
    30,
    31,
    32,
    252,
    253,
    254,
    255,
    256,
    257,
    258,
    259,
}
SK_MAX_LABEL = max(SK_LEARNING_MAP.keys()) + 100
SK_LEARNING_MAP_ARRAY = np.zeros(SK_MAX_LABEL, dtype=np.int32)
for k, v in SK_LEARNING_MAP.items():
    SK_LEARNING_MAP_ARRAY[k] = v


@dataclass
class MetricsBundle:
    class_metrics: Dict[str, Dict[str, float]]
    ratios: Dict[str, Dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Semantic-Decoupled PQ Evaluation Script (SemanticKITTI / nuScenes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["semantickitti", "nuscenes"], required=True, help="指定数据集")
    parser.add_argument("--pred-dir", type=Path, help="SemanticKITTI Predicted Label Directory (*.label)")
    parser.add_argument("--gt-dir", type=Path, help="SemanticKITTI Ground Truth Label Directory (*.label)")

    parser.add_argument("--result-path", type=Path, help="nuScenes Panoptic Predictions Root Directory")
    parser.add_argument("--dataroot", type=Path, help="nuScenes Data Root Directory")
    parser.add_argument("--version", type=str, default="v1.0-trainval", help="nuScenes Version")
    parser.add_argument("--eval-set", type=str, default="val", help="nuScenes Evaluation Split")

    parser.add_argument("--min-inst-points", type=int, default=None, help="Minimum number of instance points (nuScenes default: 15; SemanticKITTI default: 50)")
    parser.add_argument("--output", type=Path, default=None, help="Output Directory (scores.txt + detailed_results.html)")
    parser.add_argument("--verbose", action="store_true", help="Print more debugging information in nuScenes mode.")

    args = parser.parse_args()
    if args.dataset == "semantickitti":
        if args.pred_dir is None or args.gt_dir is None:
            parser.error("In SemanticKITTI mode, --pred-dir and --gt-dir must be provided.")
    else:
        missing = [flag for flag, val in [("--result-path", args.result_path), ("--dataroot", args.dataroot)] if val is None]
        if missing:
            parser.error(f"nuScenes mode is missing parameters: {', '.join(missing)}")

    if args.min_inst_points is None:
        args.min_inst_points = 50 if args.dataset == "semantickitti" else 15
    return args


def mean_metrics(class_names: List[str], metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not class_names:
        return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "IoU": 0.0}
    values = [metrics[name] for name in class_names if name in metrics]
    if not values:
        return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "IoU": 0.0}
    return {
        "PQ": float(np.mean([v["PQ"] for v in values])),
        "SQ": float(np.mean([v["SQ"] for v in values])),
        "RQ": float(np.mean([v["RQ"] for v in values])),
        "IoU": float(np.mean([v["IoU"] for v in values])),
    }


def compute_ratio(pre_metrics: Dict[str, Dict[str, float]], csi_metrics: Dict[str, Dict[str, float]], names: List[str]):
    eps = 1e-15
    ratio: Dict[str, Dict[str, float]] = {}
    for name in names:
        if name not in pre_metrics or name not in csi_metrics:
            continue
        pre = pre_metrics[name]
        csi = csi_metrics[name]
        entry = {
            "S_PQ": pre["PQ"] / max(csi["PQ"], eps),
            "S_SQ": pre["SQ"] / max(csi["SQ"], eps),
            "S_RQ": pre["RQ"] / max(csi["RQ"], eps),
            "S_IoU": pre["IoU"] / max(csi["IoU"], eps),
        }
        if "PQ_dagger" in pre and "PQ_dagger" in csi:
            entry["S_PQ_dagger"] = pre["PQ_dagger"] / max(csi["PQ_dagger"], eps)
        ratio[name] = entry
    return ratio


def evaluate_semantickitti(args: argparse.Namespace) -> MetricsBundle:
    pred_files = sorted(args.pred_dir.glob("*.label"))
    if not pred_files:
        raise FileNotFoundError(f"No *.label files found in {args.pred_dir}.")

    evaluator = PanopticEval(n_classes=20, ignore=[0], min_points=args.min_inst_points)

    def run(pass_mode: str) -> Dict[str, Dict[str, float]]:
        evaluator.reset()
        for pred_file in tqdm(pred_files, desc=f"Computing {pass_mode}"):
            gt_file = args.gt_dir / pred_file.name
            if not gt_file.exists():
                continue
            pred_label = np.fromfile(pred_file, dtype=np.uint32)
            gt_label = np.fromfile(gt_file, dtype=np.uint32)
            if len(pred_label) != len(gt_label):
                continue
            if pass_mode == "PQ_csi":
                pred_label = construct_csi_gt_semantic(pred_label, gt_label)
            pred_sem = SK_LEARNING_MAP_ARRAY[np.clip(pred_label & 0xFFFF, 0, SK_MAX_LABEL - 1)]
            pred_inst = pred_label
            gt_sem = SK_LEARNING_MAP_ARRAY[np.clip(gt_label & 0xFFFF, 0, SK_MAX_LABEL - 1)]
            gt_inst = gt_label
            evaluator.addBatch(pred_sem, pred_inst, gt_sem, gt_inst)

        pq_mean, sq_mean, rq_mean, pq_all, sq_all, rq_all = evaluator.getPQ()
        iou_mean, iou_all = evaluator.getSemIoU()
        pq_all = pq_all.flatten().tolist()
        sq_all = sq_all.flatten().tolist()
        rq_all = rq_all.flatten().tolist()
        iou_all = iou_all.flatten().tolist()

        metrics: Dict[str, Dict[str, float]] = {
            "all": {"PQ": float(pq_mean), "SQ": float(sq_mean), "RQ": float(rq_mean), "IoU": float(iou_mean)}
        }
        for idx, (pq, rq, sq, iou) in enumerate(zip(pq_all, rq_all, sq_all, iou_all)):
            class_name = SK_CLASS_STRINGS.get(idx, f"class_{idx}")
            metrics[class_name] = {"PQ": float(pq), "SQ": float(sq), "RQ": float(rq), "IoU": float(iou)}
        metrics["things"] = mean_metrics(SK_THINGS, metrics)
        metrics["stuff"] = mean_metrics(SK_STUFF, metrics)
        return metrics

    csi_metrics = run("PQ_csi")
    pre_metrics = run("PQ_pre")
    if "all" in csi_metrics:
        csi_metrics["all"]["PQ_dagger"] = float(
            np.mean(
                [csi_metrics[name]["PQ"] for name in SK_THINGS if name in csi_metrics]
                + [csi_metrics[name]["IoU"] for name in SK_STUFF if name in csi_metrics]
            )
        )
    if "all" in pre_metrics:
        pre_metrics["all"]["PQ_dagger"] = float(
            np.mean(
                [pre_metrics[name]["PQ"] for name in SK_THINGS if name in pre_metrics]
                + [pre_metrics[name]["IoU"] for name in SK_STUFF if name in pre_metrics]
            )
        )
    ratio_metrics = compute_ratio(pre_metrics, csi_metrics, SK_ALL_CLASSES + ["things", "stuff", "all"])
    return MetricsBundle(
        class_metrics={"csi": csi_metrics, "pre": pre_metrics},
        ratios={"ratio": ratio_metrics},
    )


def construct_csi_gt_semantic(pred_label: np.ndarray, gt_label: np.ndarray) -> np.ndarray:
    pred_semantics = pred_label & 0xFFFF
    gt_instances = gt_label >> 16
    pred_instances = pred_label >> 16
    mask = np.isin(pred_semantics, list(SK_TARGET_SEMANTIC_IDS))
    new_instances = np.where(mask, gt_instances, pred_instances)
    csi_gt_label = (new_instances << 16) | pred_semantics
    return csi_gt_label.astype(np.uint32)


def evaluate_nuscenes(
    args: argparse.Namespace,
    nusc,
    mapper,
    thing_names: List[str],
    stuff_names: List[str],
    thing_ids: List[int],
) -> MetricsBundle:
    from nuscenes.eval.panoptic.utils import get_samples_in_panoptic_eval_set  # type: ignore
    from nuscenes.utils.data_io import load_bin_file  # type: ignore

    ignore_idx = mapper.ignore_class["index"]
    num_classes = len(mapper.coarse_name_2_coarse_idx_mapping)
    id2name = {idx: name for name, idx in mapper.coarse_name_2_coarse_idx_mapping.items()}

    sample_tokens = get_samples_in_panoptic_eval_set(nusc, args.eval_set)
    evaluator = PanopticEval(n_classes=num_classes, ignore=[ignore_idx], min_points=args.min_inst_points)

    def run(mode: str) -> Dict[str, Dict[str, float]]:
        evaluator.reset()
        for sample_token in tqdm(sample_tokens, desc=f"Computing {mode}", disable=False):
            sample = nusc.get("sample", sample_token)
            sd_token = sample["data"]["LIDAR_TOP"]
            gt_rec = nusc.get("panoptic", sd_token)
            gt_panoptic = load_bin_file(str(Path(nusc.dataroot) / gt_rec["filename"]), type="panoptic")
            gt_sem = mapper.convert_label((gt_panoptic // 1000).astype(np.int32)).astype(np.int32)
            gt_inst = gt_panoptic
            pred_file = args.result_path / "panoptic" / args.eval_set / f"{sd_token}_panoptic.npz"
            if not pred_file.exists():
                continue
            pred_panoptic = load_bin_file(str(pred_file), type="panoptic")
            pred_sem = (pred_panoptic // 1000).astype(np.int32)
            pred_inst = pred_panoptic
            if mode == "PQ_csi":
                pred_inst = construct_csi_instances(pred_sem, pred_inst, gt_inst, thing_ids)
            evaluator.addBatch(pred_sem, pred_inst, gt_sem, gt_inst)

        pq_mean, sq_mean, rq_mean, pq_all, sq_all, rq_all = evaluator.getPQ()
        iou_mean, iou_all = evaluator.getSemIoU()
        pq_all = pq_all.flatten().tolist()
        sq_all = sq_all.flatten().tolist()
        rq_all = rq_all.flatten().tolist()
        iou_all = iou_all.flatten().tolist()

        metrics: Dict[str, Dict[str, float]] = {
            "all": {"PQ": float(pq_mean), "SQ": float(sq_mean), "RQ": float(rq_mean), "IoU": float(iou_mean)}
        }
        for idx, (pq, rq, sq, iou) in enumerate(zip(pq_all, rq_all, sq_all, iou_all)):
            class_name = id2name.get(idx, f"class_{idx}")
            metrics[class_name] = {"PQ": float(pq), "SQ": float(sq), "RQ": float(rq), "IoU": float(iou)}
        metrics["things"] = mean_metrics(thing_names, metrics)
        metrics["stuff"] = mean_metrics(stuff_names, metrics)
        metrics["all"]["PQ_dagger"] = float(
            np.mean(
                [metrics[name]["PQ"] for name in thing_names if name in metrics]
                + [metrics[name]["IoU"] for name in stuff_names if name in metrics]
            )
        )
        return metrics

    csi_metrics = run("PQ_csi")
    pre_metrics = run("PQ_pre")
    ratio_metrics = compute_ratio(pre_metrics, csi_metrics, thing_names + stuff_names + ["things", "stuff", "all"])
    return MetricsBundle(
        class_metrics={"csi": csi_metrics, "pre": pre_metrics},
        ratios={"ratio": ratio_metrics},
    )


def construct_csi_instances(pred_sem: np.ndarray, pred_inst: np.ndarray, gt_inst: np.ndarray, thing_ids: Sequence[int]):
    mask = np.isin(pred_sem, thing_ids)
    if not np.any(mask):
        return pred_inst
    new_inst = pred_inst.copy()
    new_inst[mask] = gt_inst[mask]
    return new_inst


def print_metrics_table(metrics: Dict[str, Dict[str, float]], thing_names: List[str], stuff_names: List[str], title: str):
    print(f"\n{'=' * 90}")
    print(title)
    print(f"{'=' * 90}")
    print(f"{'Class':<20} | {'PQ':<12} | {'SQ':<12} | {'RQ':<12} | {'IoU':<12}")
    print("-" * 90)

    print("--- Things ---")
    for name in thing_names:
        entry = metrics.get(name, {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "IoU": 0.0})
        print(f"{name:<20} | {entry['PQ']:.6f}     | {entry['SQ']:.6f}     | {entry['RQ']:.6f}     | {entry['IoU']:.6f}")
    print("-" * 90)
    entry = metrics["things"]
    print(f"{'Mean (Things)':<20} | {entry['PQ']:.6f}     | {entry['SQ']:.6f}     | {entry['RQ']:.6f}     | {entry['IoU']:.6f}")

    print("-" * 90)
    print("--- Stuff ---")
    for name in stuff_names:
        entry = metrics.get(name, {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "IoU": 0.0})
        print(f"{name:<20} | {entry['PQ']:.6f}     | {entry['SQ']:.6f}     | {entry['RQ']:.6f}     | {entry['IoU']:.6f}")
    print("-" * 90)
    entry = metrics["stuff"]
    print(f"{'Mean (Stuff)':<20} | {entry['PQ']:.6f}     | {entry['SQ']:.6f}     | {entry['RQ']:.6f}     | {entry['IoU']:.6f}")

    print("-" * 90)
    entry = metrics["all"]
    print(f"{'Mean (All)':<20} | {entry['PQ']:.6f}     | {entry['SQ']:.6f}     | {entry['RQ']:.6f}     | {entry['IoU']:.6f}")
    if "PQ_dagger" in entry:
        print(f"{'PQ_dagger (All)':<20} | {entry['PQ_dagger']:.6f}")
    print("=" * 90)


def print_ratio_table(ratio: Dict[str, Dict[str, float]], thing_names: List[str], stuff_names: List[str], title: str):
    print(f"\n{'=' * 90}")
    print(title)
    print(f"{'=' * 90}")
    print(f"{'Class':<20} | {'S_PQ':<12} | {'S_SQ':<12} | {'S_RQ':<12} | {'S_IoU':<12}")
    print("-" * 90)
    print("--- Things ---")
    for name in thing_names:
        entry = ratio.get(name, {"S_PQ": 0.0, "S_SQ": 0.0, "S_RQ": 0.0, "S_IoU": 0.0})
        print(f"{name:<20} | {entry['S_PQ']:.6f}     | {entry['S_SQ']:.6f}     | {entry['S_RQ']:.6f}     | {entry['S_IoU']:.6f}")
    print("-" * 90)
    entry = ratio.get("things", {"S_PQ": 0.0, "S_SQ": 0.0, "S_RQ": 0.0, "S_IoU": 0.0})
    print(f"{'Mean (Things)':<20} | {entry['S_PQ']:.6f}     | {entry['S_SQ']:.6f}     | {entry['S_RQ']:.6f}     | {entry['S_IoU']:.6f}")

    print("-" * 90)
    print("--- Stuff ---")
    for name in stuff_names:
        entry = ratio.get(name, {"S_PQ": 0.0, "S_SQ": 0.0, "S_RQ": 0.0, "S_IoU": 0.0})
        print(f"{name:<20} | {entry['S_PQ']:.6f}     | {entry['S_SQ']:.6f}     | {entry['S_RQ']:.6f}     | {entry['S_IoU']:.6f}")
    print("-" * 90)
    entry = ratio.get("stuff", {"S_PQ": 0.0, "S_SQ": 0.0, "S_RQ": 0.0, "S_IoU": 0.0})
    print(f"{'Mean (Stuff)':<20} | {entry['S_PQ']:.6f}     | {entry['S_SQ']:.6f}     | {entry['S_RQ']:.6f}     | {entry['S_IoU']:.6f}")

    print("-" * 90)
    entry = ratio.get("all", {"S_PQ": 0.0, "S_SQ": 0.0, "S_RQ": 0.0, "S_IoU": 0.0})
    print(f"{'Mean (All)':<20} | {entry['S_PQ']:.6f}     | {entry['S_SQ']:.6f}     | {entry['S_RQ']:.6f}     | {entry['S_IoU']:.6f}")
    if "S_PQ_dagger" in entry:
        print(f"{'S_PQ_dagger (All)':<20} | {entry['S_PQ_dagger']:.6f}")
    print("=" * 90)


def save_outputs(output_dir: Path, thing_names: List[str], stuff_names: List[str], bundle: MetricsBundle):
    output_dir.mkdir(parents=True, exist_ok=True)
    csi = bundle.class_metrics["csi"]
    pre = bundle.class_metrics["pre"]
    ratio = bundle.ratios["ratio"]
    csi_metrics = csi
    pre_metrics = pre
    ratio_metrics = ratio

    data = {
        "csi_metrics": {
            "pq_things": csi["things"]["PQ"],
            "sq_things": csi["things"]["SQ"],
            "rq_things": csi["things"]["RQ"],
            "iou_things": csi["things"]["IoU"],
            "pq_stuff": csi["stuff"]["PQ"],
            "sq_stuff": csi["stuff"]["SQ"],
            "rq_stuff": csi["stuff"]["RQ"],
            "iou_stuff": csi["stuff"]["IoU"],
            "pq_all": csi["all"]["PQ"],
            "sq_all": csi["all"]["SQ"],
            "rq_all": csi["all"]["RQ"],
            "iou_all": csi["all"]["IoU"],
            "pq_dagger_all": csi["all"].get("PQ_dagger", 0.0),
        },
        "pre_metrics": {
            "pq_things": pre["things"]["PQ"],
            "sq_things": pre["things"]["SQ"],
            "rq_things": pre["things"]["RQ"],
            "iou_things": pre["things"]["IoU"],
            "pq_stuff": pre["stuff"]["PQ"],
            "sq_stuff": pre["stuff"]["SQ"],
            "rq_stuff": pre["stuff"]["RQ"],
            "iou_stuff": pre["stuff"]["IoU"],
            "pq_all": pre["all"]["PQ"],
            "sq_all": pre["all"]["SQ"],
            "rq_all": pre["all"]["RQ"],
            "iou_all": pre["all"]["IoU"],
            "pq_dagger_all": pre["all"].get("PQ_dagger", 0.0),
        },
        "ratio_metrics": {
            "s_pq_things": ratio.get("things", {}).get("S_PQ", 0.0),
            "s_sq_things": ratio.get("things", {}).get("S_SQ", 0.0),
            "s_rq_things": ratio.get("things", {}).get("S_RQ", 0.0),
            "s_iou_things": ratio.get("things", {}).get("S_IoU", 0.0),
            "s_pq_stuff": ratio.get("stuff", {}).get("S_PQ", 0.0),
            "s_sq_stuff": ratio.get("stuff", {}).get("S_SQ", 0.0),
            "s_rq_stuff": ratio.get("stuff", {}).get("S_RQ", 0.0),
            "s_iou_stuff": ratio.get("stuff", {}).get("S_IoU", 0.0),
            "s_pq_all": ratio.get("all", {}).get("S_PQ", 0.0),
            "s_sq_all": ratio.get("all", {}).get("S_SQ", 0.0),
            "s_rq_all": ratio.get("all", {}).get("S_RQ", 0.0),
            "s_iou_all": ratio.get("all", {}).get("S_IoU", 0.0),
            "s_pq_dagger_all": ratio.get("all", {}).get("S_PQ_dagger", 0.0),
        },
    }

    all_names = thing_names + stuff_names
    for name in all_names:
        data["csi_metrics"][f"pq_{name}"] = csi[name]["PQ"]
        data["csi_metrics"][f"sq_{name}"] = csi[name]["SQ"]
        data["csi_metrics"][f"rq_{name}"] = csi[name]["RQ"]
        data["csi_metrics"][f"iou_{name}"] = csi[name]["IoU"]

        data["pre_metrics"][f"pq_{name}"] = pre[name]["PQ"]
        data["pre_metrics"][f"sq_{name}"] = pre[name]["SQ"]
        data["pre_metrics"][f"rq_{name}"] = pre[name]["RQ"]
        data["pre_metrics"][f"iou_{name}"] = pre[name]["IoU"]

        data["ratio_metrics"][f"s_pq_{name}"] = ratio[name]["S_PQ"]
        data["ratio_metrics"][f"s_sq_{name}"] = ratio[name]["S_SQ"]
        data["ratio_metrics"][f"s_rq_{name}"] = ratio[name]["S_RQ"]
        data["ratio_metrics"][f"s_iou_{name}"] = ratio[name]["S_IoU"]

    scores_file = output_dir / "scores.txt"
    with open(scores_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    print(f"Results saved to: {scores_file}")

    table_csi, table_pre, table_ratio = [], [], []
    for name in all_names:
        t = "things" if name in thing_names else "stuff"
        table_csi.append(
            {"class": name, "type": t, "pq": f"{csi[name]['PQ']:.6f}", "sq": f"{csi[name]['SQ']:.6f}", "rq": f"{csi[name]['RQ']:.6f}", "iou": f"{csi[name]['IoU']:.6f}"}
        )
        table_pre.append(
            {"class": name, "type": t, "pq": f"{pre[name]['PQ']:.6f}", "sq": f"{pre[name]['SQ']:.6f}", "rq": f"{pre[name]['RQ']:.6f}", "iou": f"{pre[name]['IoU']:.6f}"}
        )
        ratio_entry = ratio.get(name, {"S_PQ": 0.0, "S_SQ": 0.0, "S_RQ": 0.0, "S_IoU": 0.0})
        table_ratio.append(
            {
                "class": name,
                "type": t,
                "s_pq": f"{ratio_entry['S_PQ']:.6f}",
                "s_sq": f"{ratio_entry['S_SQ']:.6f}",
                "s_rq": f"{ratio_entry['S_RQ']:.6f}",
                "s_iou": f"{ratio_entry['S_IoU']:.6f}",
            }
        )

    html_file = output_dir / "detailed_results.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(
            """<!doctype html>
<html lang="en" style="scroll-behavior: smooth;">
<head>
  <meta charset="UTF-8">
  <title>Unified S-Decoupled PQ Results</title>
  <script src='https://cdnjs.cloudflare.com/ajax/libs/tabulator/4.4.3/js/tabulator.min.js'></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tabulator/4.4.3/css/bulma/tabulator_bulma.min.css">
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    h1, h2 { color: #333; }
    .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    .summary h2 { margin-top: 0; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 10px; }
    .card { background: #e8e8e8; padding: 10px; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>Unified Semantic-Decoupled PQ Results</h1>
  <div class="summary">
    <h2>Ratio Summary (S = PQ_pre / PQ_csi)</h2>
    <div class="grid">
      <div class="card">
        <h3>Things</h3>
        <p><strong>S_PQ:</strong> """
            + f"{ratio['things']['S_PQ']:.10f}"
            + """</p>
        <p><strong>S_SQ:</strong> """
            + f"{ratio['things']['S_SQ']:.10f}"
            + """</p>
        <p><strong>S_RQ:</strong> """
            + f"{ratio['things']['S_RQ']:.10f}"
            + """</p>
        <p><strong>S_IoU:</strong> """
            + f"{ratio['things']['S_IoU']:.10f}"
            + """</p>
      </div>
      <div class="card">
        <h3>Stuff</h3>
        <p><strong>S_PQ:</strong> """
            + f"{ratio['stuff']['S_PQ']:.10f}"
            + """</p>
        <p><strong>S_SQ:</strong> """
            + f"{ratio['stuff']['S_SQ']:.10f}"
            + """</p>
        <p><strong>S_RQ:</strong> """
            + f"{ratio['stuff']['S_RQ']:.10f}"
            + """</p>
        <p><strong>S_IoU:</strong> """
            + f"{ratio['stuff']['S_IoU']:.10f}"
            + """</p>
      </div>
      <div class="card">
        <h3>All</h3>
        <p><strong>S_PQ:</strong> """
            + f"{ratio['all']['S_PQ']:.10f}"
            + """</p>
        <p><strong>S_SQ:</strong> """
            + f"{ratio['all']['S_SQ']:.10f}"
            + """</p>
        <p><strong>S_RQ:</strong> """
            + f"{ratio['all']['S_RQ']:.10f}"
            + """</p>
        <p><strong>S_IoU:</strong> """
            + f"{ratio['all']['S_IoU']:.10f}"
            + """</p>
        <p><strong>S_PQ_dagger:</strong> """
            + f"{ratio['all'].get('S_PQ_dagger', 0.0):.10f}"
            + """</p>
      </div>
    </div>
  </div>

  <h2>CSI-GT Results (PQ_csi) - Theoretical Upper Bound of Panoptic Segmentation</h2>
  <p>PQ Things: """ + f"{csi_metrics['things']['PQ']:.10f}" + """ | PQ Stuff: """ + f"{csi_metrics['stuff']['PQ']:.10f}" + """ | PQ All: """ + f"{csi_metrics['all']['PQ']:.10f}" + """ | PQ_dagger All: """ + f"{csi_metrics['all'].get('PQ_dagger', 0.0):.10f}" + """ | mIoU All: """ + f"{csi_metrics['all']['IoU']:.10f}" + """</p>
  <p>RQ Things: """ + f"{csi_metrics['things']['RQ']:.10f}" + """ | RQ Stuff: """ + f"{csi_metrics['stuff']['RQ']:.10f}" + """ | RQ All: """ + f"{csi_metrics['all']['RQ']:.10f}" + """</p>
  <p>SQ Things: """ + f"{csi_metrics['things']['SQ']:.10f}" + """ | SQ Stuff: """ + f"{csi_metrics['stuff']['SQ']:.10f}" + """ | SQ All: """ + f"{csi_metrics['all']['SQ']:.10f}" + """</p>
  <div id="table_csi"></div>

  <h2>Prediction Results (PQ_pre) - Actual Prediction Performance</h2>
  <p>PQ Things: """ + f"{pre_metrics['things']['PQ']:.10f}" + """ | PQ Stuff: """ + f"{pre_metrics['stuff']['PQ']:.10f}" + """ | PQ All: """ + f"{pre_metrics['all']['PQ']:.10f}" + """ | PQ_dagger All: """ + f"{pre_metrics['all'].get('PQ_dagger', 0.0):.10f}" + """ | mIoU All: """ + f"{pre_metrics['all']['IoU']:.10f}" + """</p>
  <p>RQ Things: """ + f"{pre_metrics['things']['RQ']:.10f}" + """ | RQ Stuff: """ + f"{pre_metrics['stuff']['RQ']:.10f}" + """ | RQ All: """ + f"{pre_metrics['all']['RQ']:.10f}" + """</p>
  <p>SQ Things: """ + f"{pre_metrics['things']['SQ']:.10f}" + """ | SQ Stuff: """ + f"{pre_metrics['stuff']['SQ']:.10f}" + """ | SQ All: """ + f"{pre_metrics['all']['SQ']:.10f}" + """</p>
  <div id="table_pre"></div>

  <h2>Ratio Results (S = PQ_pre / PQ_csi)</h2>
  <p>S_PQ Things: """ + f"{ratio['things']['S_PQ']:.10f}" + """ | S_PQ Stuff: """ + f"{ratio['stuff']['S_PQ']:.10f}" + """ | S_PQ All: """ + f"{ratio['all']['S_PQ']:.10f}" + """ | S_PQ_dagger All: """ + f"{ratio['all'].get('S_PQ_dagger', 0.0):.10f}" + """ | mIoU All: """ + f"{ratio['all']['S_IoU']:.10f}" + """</p>
  <p>S_RQ Things: """ + f"{ratio['things']['S_RQ']:.10f}" + """ | S_RQ Stuff: """ + f"{ratio['stuff']['S_RQ']:.10f}" + """ | S_RQ All: """ + f"{ratio['all']['S_RQ']:.10f}" + """</p>
  <p>S_SQ Things: """ + f"{ratio['things']['S_SQ']:.10f}" + """ | S_SQ Stuff: """ + f"{ratio['stuff']['S_SQ']:.10f}" + """ | S_SQ All: """ + f"{ratio['all']['S_SQ']:.10f}" + """</p>
  <div id="table_ratio"></div>

<script>
  let table_csi_data = """
            + json.dumps(table_csi)
            + """;
  let table_pre_data = """
            + json.dumps(table_pre)
            + """;
  let table_ratio_data = """
            + json.dumps(table_ratio)
            + """;

  function createTable(id, data, columns) {
    new Tabulator(id, {
      layout: "fitColumns",
      data: data,
      columns: columns
    });
  }

  const columns_base = [
    {title: "Class", field:"class", width:180},
    {title: "Type", field:"type", width:100, align: "center"}
  ];

  createTable("#table_csi", table_csi_data, columns_base.concat([
    {title: "PQ", field:"pq", width:120, align: "center"},
    {title: "SQ", field:"sq", width:120, align: "center"},
    {title: "RQ", field:"rq", width:120, align: "center"},
    {title: "IoU", field:"iou", width:120, align: "center"}
  ]));

  createTable("#table_pre", table_pre_data, columns_base.concat([
    {title: "PQ", field:"pq", width:120, align: "center"},
    {title: "SQ", field:"sq", width:120, align: "center"},
    {title: "RQ", field:"rq", width:120, align: "center"},
    {title: "IoU", field:"iou", width:120, align: "center"}
  ]));

  createTable("#table_ratio", table_ratio_data, columns_base.concat([
    {title: "S_PQ", field:"s_pq", width:120, align: "center"},
    {title: "S_SQ", field:"s_sq", width:120, align: "center"},
    {title: "S_RQ", field:"s_rq", width:120, align: "center"},
    {title: "S_IoU", field:"s_iou", width:120, align: "center"}
  ]));
</script>
</body>
</html>"""
        )
    print(f"Detailed results have been saved to: {html_file}")


def main():
    args = parse_args()
    start_time = time.time()

    if args.dataset == "semantickitti":
        print("*" * 90)
        print("SemanticKITTI Semantic-Decoupled PQ Evaluation")
        print("*" * 90)
        print(f"Prediction Directory: {args.pred_dir}")
        print(f"Ground Truth Directory: {args.gt_dir}")
        print(f"Minimum Instance Points: {args.min_inst_points}")
        print(f"Output Directory: {args.output}")
        print("*" * 90)
        bundle = evaluate_semantickitti(args)
        thing_names = SK_THINGS
        stuff_names = SK_STUFF
    else:
        print("*" * 90)
        print("nuScenes Semantic-Decoupled PQ Evaluation")
        print("*" * 90)
        print(f"Prediction Directory: {args.result_path}")
        print(f"Data Root Directory: {args.dataroot}")
        print(f"Version: {args.version}")
        print(f"Evaluation split: {args.eval_set}")
        print(f"Minimum Instance Points: {args.min_inst_points}")
        print(f"Output Directory: {args.output}")
        print("*" * 90)
        from nuscenes.eval.panoptic.utils import PanopticClassMapper  # type: ignore
        from nuscenes.nuscenes import NuScenes  # type: ignore

        nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=args.verbose)
        mapper = PanopticClassMapper(nusc)
        thing_names = [name for name, _ in sorted(mapper.things.items(), key=lambda x: x[1])]
        stuff_names = [name for name, _ in sorted(mapper.stuff.items(), key=lambda x: x[1])]
        thing_ids = [mapper.things[name] for name in thing_names]
        bundle = evaluate_nuscenes(args, nusc, mapper, thing_names, stuff_names, thing_ids)

    elapsed = time.time() - start_time
    print(f"\nCompleted {args.dataset} S-PQ evaluation. Total time elapsed: {elapsed:.2f} seconds.")
    print_metrics_table(bundle.class_metrics["csi"], thing_names, stuff_names, "CSI-GT Results (PQ_csi)")
    print_metrics_table(bundle.class_metrics["pre"], thing_names, stuff_names, "Prediction Results (PQ_pre)")
    print_ratio_table(bundle.ratios["ratio"], thing_names, stuff_names, "S-Decoupled PQ Results (S)")

    if args.output is not None:
        save_outputs(args.output, thing_names, stuff_names, bundle)


if __name__ == "__main__":
    main()
