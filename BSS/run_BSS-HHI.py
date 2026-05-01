import argparse
import contextlib
import csv
import importlib.util
import io
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
TARGET_MODULE = THIS_DIR / "BSS-HHI_main.py"


DATASET_RUN_CONFIGS = {
    "kosarak": {
        "b": 16,
        "iterations": 5,
        "case": {
            "topk": [16, 32],
            "epsilon": [3],
            "ratios": [0.1, 0.1, 0.8],
            "search_mode": "full",
            "max_split_order": 3,
            "significant_bits_by_depth": "2,1,1",
            "user_ratio_by_depth": "0.5,0.35,0.7",
            "conditional_candidate_k_by_depth": "50,35,25,15,10",
            "global_candidate_k": 2000,
            "stage2_topd_ratio": [0.30],
        },
    },
    "movielens": {
        "b": 18,
        "iterations": 5,
        "case": {
            "topk": [32],
            "epsilon": [4],
            "ratios": [0.1, 0.1, 0.8],
            "search_mode": "global_only",
            "max_split_order": 1,
            "significant_bits_by_depth": "global_only",
            "user_ratio_by_depth": "global_only",
            "conditional_candidate_k_by_depth": "0",
            "global_candidate_k": 42025,
            "global_only_candidate_k": 42025,
            "stage2_topd_ratio": [0.1],
        },
    },
    "userbehavior1300": {
        "b": 20,
        "iterations": 5,
        "case": {
            "topk": [16, 32],
            "epsilon": [3],
            "ratios": [0.1, 0.1, 0.8],
            "search_mode": "full",
            "max_split_order": 4,
            "significant_bits_by_depth": "6,5,3,1",
            "user_ratio_by_depth": "0.5,0.35,0.55,0.75",
            "conditional_candidate_k_by_depth": "50,35,25,15,10",
            "global_candidate_k": 4000,
            "stage2_topd_ratio": [0.7],
        },
    },
    "syn65k": {
        "b": 17,
        "iterations": 5,
        "case": {
            "topk": [16, 32],
            "epsilon": [3],
            "ratios": [0.1, 0.1, 0.8],
            "search_mode": "full",
            "max_split_order": 2,
            "significant_bits_by_depth": "6,2",
            "user_ratio_by_depth": "0.5,0.45",
            "conditional_candidate_k_by_depth": "50,35,25,15,10",
            "global_candidate_k": 2000,
            "stage2_topd_ratio": [0.7],
        },
    },
    "syn130k": {
        "b": 18,
        "iterations": 5,
        "case": {
            "topk": [16, 32],
            "epsilon": [3],
            "ratios": [0.1, 0.1, 0.8],
            "search_mode": "global_only",
            "max_split_order": 1,
            "significant_bits_by_depth": "global_only",
            "user_ratio_by_depth": "global_only",
            "conditional_candidate_k_by_depth": "0",
            "global_candidate_k": 13108,
            "global_only_candidate_k": 13108,
            "stage2_topd_ratio": [0.5],
        },
    },
}


def load_target_module():
    spec = importlib.util.spec_from_file_location("BSS_HHI_main", TARGET_MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def dataset_output_tag(dataname: str) -> str:
    return dataname


def normalize_dataset_name(dataname: str) -> str:
    key = dataname.lower().strip()
    if key == "syn131k":
        return "syn130k"
    return key


def make_case(
    topk,
    ratios,
    search_mode,
    max_split_order,
    significant_bits_by_depth,
    user_ratio_by_depth,
    conditional_candidate_k_by_depth,
    global_candidate_k,
    stage2_topd_ratio,
    global_only_candidate_k=None,
    epsilon=None,
):
    return {
        "topk": topk,
        "epsilon": epsilon,
        "ratios": list(ratios),
        "search_mode": str(search_mode),
        "max_split_order": max_split_order,
        "significant_bits_by_depth": significant_bits_by_depth,
        "user_ratio_by_depth": user_ratio_by_depth,
        "conditional_candidate_k_by_depth": conditional_candidate_k_by_depth,
        "global_candidate_k": int(global_candidate_k),
        "global_only_candidate_k": (
            int(global_only_candidate_k) if global_only_candidate_k is not None else int(global_candidate_k)
        ),
        "stage2_topd_ratio": stage2_topd_ratio,
    }


def curated_cases(dataname: str):
    dataset_key = normalize_dataset_name(dataname)
    if dataset_key not in DATASET_RUN_CONFIGS:
        known = ", ".join(sorted(DATASET_RUN_CONFIGS.keys()))
        raise ValueError(f"Unsupported dataname '{dataname}'. Supported datasets: {known}")
    return [make_case(**DATASET_RUN_CONFIGS[dataset_key]["case"])]


def parse_args():
    parser = argparse.ArgumentParser(description="Run curated full BSS-HHI experiments.")
    parser.add_argument("--dataname", type=str, default="kosarak")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--epsilon", type=float, nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--base-seed", dest="base_seed", type=int, default=20260315)
    parser.add_argument("--b", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--theta", type=float, default=0.9)
    parser.add_argument("--global-user-ratio", dest="global_user_ratio", type=float, default=0.5)
    parser.add_argument("--node-user-ratio", dest="node_user_ratio", type=float, default=0.5)
    parser.add_argument("--fixed-bit-one-ratio", dest="fixed_bit_one_ratio", type=float, default=0.45)
    parser.add_argument("--soft-margin-log", dest="soft_margin_log", type=float, default=0.05)
    parser.add_argument("--soft-search-factor", dest="soft_search_factor", type=int, default=2)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    return parser.parse_args()


@dataclass
class RunResult:
    dataname: str
    topk: int
    epsilon: float
    repeat_id: int
    seed: int
    search_mode: str
    ratio_group1: float
    ratio_group2: float
    ratio_group3: float
    max_split_order: int
    significant_bits_by_depth: str
    user_ratio_by_depth: str
    conditional_candidate_k_by_depth: str
    global_candidate_k: int
    effective_root_candidate_k: int
    stage2_topd_ratio: float
    f1: float
    ncr: float
    mcr: float


class Tee:
    def __init__(self, *targets):
        self.targets = targets

    def write(self, data):
        for target in self.targets:
            target.write(data)
        return len(data)

    def flush(self):
        for target in self.targets:
            target.flush()


def emit(text, log_file):
    print(text)
    print(text, file=log_file)


def unique_preserve_order(values):
    return list(dict.fromkeys(values))


def normalize_stage2_topd_ratios(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def normalize_case_values(value, cast_type):
    if isinstance(value, (list, tuple)):
        return [cast_type(v) for v in value]
    return [cast_type(value)]


def expand_cases(cases, default_epsilons):
    expanded = []
    for case in cases:
        topk_values = normalize_case_values(case["topk"], int)
        epsilon_source = case["epsilon"] if case.get("epsilon") is not None else default_epsilons
        epsilon_values = normalize_case_values(epsilon_source, float)
        for topk in topk_values:
            for epsilon in epsilon_values:
                for ratio in normalize_stage2_topd_ratios(case["stage2_topd_ratio"]):
                    expanded_case = dict(case)
                    expanded_case["topk"] = int(topk)
                    expanded_case["epsilon"] = float(epsilon)
                    expanded_case["stage2_topd_ratio"] = float(ratio)
                    expanded.append(expanded_case)
    return expanded


def build_stage1_union_with_global_scores(bss, stage1_debug, b, search_mode):
    if search_mode == "global_only":
        stage1_union = unique_preserve_order(stage1_debug["global_decimal_candidates"])
    else:
        stage1_union = unique_preserve_order(bss.collect_stage1_candidate_union(stage1_debug))
    global_bit_freq = np.array(stage1_debug["global_bit_freq"], dtype=float)
    stage1_scores = [
        float(bss.compute_binary_probability(item_id, global_bit_freq, b, fixed_bits={}))
        for item_id in stage1_union
    ]
    ranked = sorted(zip(stage1_union, stage1_scores), key=lambda x: (-x[1], x[0]))
    item_ids = [item for item, _ in ranked]
    item_scores = [score for _, score in ranked]
    return item_ids, item_scores


def build_dynamic_stage2_inputs(stage1_item_ids, stage1_item_scores, pad_id):
    candidate_size = len(stage1_item_ids)
    matrix_side = max(1, int(math.ceil(math.sqrt(max(candidate_size, 1)))))
    padded_size = matrix_side * matrix_side
    pad_count = padded_size - candidate_size
    padded_ids = list(stage1_item_ids) + [pad_id] * pad_count
    padded_scores = list(stage1_item_scores) + [0.0] * pad_count
    s0 = np.array(padded_scores, dtype=float).reshape(matrix_side, matrix_side)
    return s0, padded_ids, matrix_side


def recover_item_count_dict_from_top_list(mapped_item_dict, top_d_list):
    reverse_mapping = {new_id: original_id for new_id, original_id in enumerate(top_d_list)}
    recovered = {}
    for key, value in mapped_item_dict.items():
        mapped_id = key[0]
        if mapped_id in reverse_mapping:
            recovered[(reverse_mapping[mapped_id],)] = value
    return recovered


def summarize_numeric_values(values):
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "avg": 0.0,
            "min": 0,
            "max": 0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "count": int(arr.size),
        "avg": float(np.mean(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def bucketize_values(values, bucket_edges):
    buckets = Counter()
    for val in values:
        placed = False
        for label, low, high in bucket_edges:
            if low <= val <= high:
                buckets[label] += 1
                placed = True
                break
        if not placed:
            buckets[f">{bucket_edges[-1][2]}"] += 1
    ordered = {}
    for label, _, _ in bucket_edges:
        ordered[label] = int(buckets.get(label, 0))
    tail_label = f">{bucket_edges[-1][2]}"
    if tail_label in buckets:
        ordered[tail_label] = int(buckets[tail_label])
    return ordered


def format_summary(prefix, summary):
    return (
        f"{prefix}=(count={summary['count']}, avg={summary['avg']:.4f}, min={summary['min']}, "
        f"max={summary['max']}, median={summary['median']:.2f}, p75={summary['p75']:.2f}, "
        f"p90={summary['p90']:.2f}, p95={summary['p95']:.2f}, p99={summary['p99']:.2f})"
    )


def ranked_real_topk(module, data, topk):
    metrics_itemk = module.MetricsItem(data, top_k=topk)
    real_topk_item = metrics_itemk.true_item_dict
    real_topk_ranked = sorted(
        [(item_tuple[0], freq) for item_tuple, freq in real_topk_item.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    return metrics_itemk, real_topk_ranked, {item for item, _ in real_topk_ranked}


def run_one(module, args, case, repeat_id, log_file):
    seed = args.base_seed + repeat_id
    random.seed(seed)
    np.random.seed(seed)

    domain_size, user_total = module.resolve_dataset_meta(args.dataname, None, None)
    data = module.Data(
        dataname=args.dataname,
        limit=args.limit,
        domain_size=domain_size,
        user_total=user_total,
    )
    bss = module.BSSHHI(data)
    metrics_itemk, real_topk_ranked, _ = ranked_real_topk(module, data, case["topk"])

    group1, group2, group3 = bss.split_data_by_ratio(data, case["ratios"])

    if case["search_mode"] == "global_only":
        sig_by_depth = [1]
        user_ratio_by_depth = None
        conditional_candidate_k_by_depth = [1]
        effective_root_candidate_k = int(case["global_only_candidate_k"])
    else:
        sig_by_depth = module.parse_depth_config(case["significant_bits_by_depth"], cast_type=int)
        user_ratio_by_depth = module.parse_depth_config(case["user_ratio_by_depth"], cast_type=float)
        conditional_candidate_k_by_depth = module.parse_depth_config(
            case["conditional_candidate_k_by_depth"], cast_type=int
        )
        effective_root_candidate_k = int(case["global_candidate_k"])

    legacy_side = max(1, int(np.ceil(np.sqrt(max(effective_root_candidate_k, 1)))))
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, stage1_debug = bss.init_matrix(
            group1=group1,
            epsilon=case["epsilon"],
            topk=legacy_side,
            b=args.b,
            significant_bit_count=2,
            global_user_ratio=args.global_user_ratio,
            fixed_bit_one_ratio=args.fixed_bit_one_ratio,
            global_candidate_k=effective_root_candidate_k,
            conditional_candidate_k=conditional_candidate_k_by_depth[0],
            conditional_candidate_k_by_depth=conditional_candidate_k_by_depth,
            max_split_order=case["max_split_order"],
            node_user_ratio=args.node_user_ratio,
            significant_bits_by_depth=sig_by_depth,
            user_ratio_by_depth=user_ratio_by_depth,
            soft_margin_log=args.soft_margin_log,
            soft_search_factor=args.soft_search_factor,
            return_debug=True,
        )

    stage1_item_ids, stage1_item_scores = build_stage1_union_with_global_scores(
        bss, stage1_debug, args.b, case["search_mode"]
    )
    stage1_union = list(stage1_item_ids)
    stage1_union_set = set(stage1_union)
    stage1_hit_ids = [item for item, _ in real_topk_ranked if item in stage1_union_set]
    stage1_missed_ids = [item for item, _ in real_topk_ranked if item not in stage1_union_set]

    pad_id = int(group2.dict_size + 1)
    s0_dynamic, k_list_dynamic, matrix_side = build_dynamic_stage2_inputs(
        stage1_item_ids,
        stage1_item_scores,
        pad_id=pad_id,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        s_iter = bss.svd_iteration(
            s0_dynamic,
            group2.data,
            k_list_dynamic,
            d=matrix_side,
            domain_size=group2.dict_size,
            iterations=args.iterations,
            epsilon=case["epsilon"],
            theta=args.theta,
        )

    raw_top_d = max(int(np.ceil(case["stage2_topd_ratio"] * len(stage1_union))), int(case["topk"]))
    raw_top_d = min(raw_top_d, len(stage1_union))
    stage2_candidates = bss.get_topd_original_indices(
        s_iter,
        k_list_dynamic,
        raw_top_d,
        invalid_ids={pad_id},
    )
    stage2_candidates = unique_preserve_order(stage2_candidates)
    stage2_candidate_set = set(stage2_candidates)
    stage2_hit_ids = [item for item, _ in real_topk_ranked if item in stage2_candidate_set]
    stage2_missed_ids = [item for item, _ in real_topk_ranked if item not in stage2_candidate_set]

    set_size_buckets = [
        ("0", 0, 0),
        ("1-5", 1, 5),
        ("6-10", 6, 10),
        ("11-20", 11, 20),
        ("21-50", 21, 50),
        ("51-100", 51, 100),
        ("101-200", 101, 200),
    ]
    item_freq_buckets = [
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-10", 6, 10),
        ("11-20", 11, 20),
        ("21-50", 21, 50),
        ("51-100", 51, 100),
        ("101-500", 101, 500),
        ("501-1000", 501, 1000),
        ("1001-5000", 1001, 5000),
    ]

    group3_user_count_before = len(group3.data)
    group3_lengths_before = [len(items) for items in group3.data]
    group3_stats_before = summarize_numeric_values(group3_lengths_before)
    group3_buckets_before = bucketize_values(group3_lengths_before, set_size_buckets)
    group3, user_num = bss.remove_non_top_items(group3, stage2_candidates)
    group3_user_count_after = len(group3.data)
    group3_lengths_after = [len(items) for items in group3.data]
    group3_stats_after = summarize_numeric_values(group3_lengths_after)
    group3_buckets_after = bucketize_values(group3_lengths_after, set_size_buckets)

    item_counter_after = Counter()
    for items in group3.data:
        item_counter_after.update(items)
    item_freq_values_after = list(item_counter_after.values())
    item_freq_stats_after = summarize_numeric_values(item_freq_values_after)
    item_freq_buckets_after = bucketize_values(item_freq_values_after, item_freq_buckets)

    stage3_metrics_itemk = module.MetricsItem(group3, top_k=case["topk"])
    local_true_topk_recovered = recover_item_count_dict_from_top_list(
        stage3_metrics_itemk.true_item_dict,
        stage2_candidates,
    )
    local_true_topk_ranked = sorted(
        [(item_tuple[0], freq) for item_tuple, freq in local_true_topk_recovered.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    local_true_topk_set = {item for item, _ in local_true_topk_ranked}
    global_true_topk_set = {item for item, _ in real_topk_ranked}
    local_global_overlap = [item for item, _ in real_topk_ranked if item in local_true_topk_set]

    svim = module.SVIM(group3, top_k=case["topk"], epsilon=case["epsilon"])
    with contextlib.redirect_stdout(io.StringIO()):
        estimate_list_svim, estimate_freq_svim = svim.find_singleton_origin(user_num)

    estimate_list_svim = bss.recover_items_from_top_list(estimate_list_svim, stage2_candidates)
    estimate_dict_svim = svim.convert_result2dict(estimate_list_svim, estimate_freq_svim)

    pred_ranked = [(item_tuple[0], freq) for item_tuple, freq in estimate_dict_svim.items()]
    pred_item_order = list(estimate_dict_svim.keys())
    stage3_hit_ids = [item for item, _ in real_topk_ranked if (item,) in pred_item_order]
    stage3_missed_ids = [item for item, _ in real_topk_ranked if (item,) not in pred_item_order]

    f1 = metrics_itemk.item_f1(estimate_dict_svim)
    ncr = metrics_itemk.item_ncr(estimate_dict_svim)
    mcr = metrics_itemk.cal_mcr(estimate_dict_svim)

    true_item_order = list(metrics_itemk.true_item_dict.keys())
    tp = len(set(true_item_order) & set(pred_item_order))
    ncr_hits = []
    for item in pred_item_order[:len(true_item_order)]:
        if item in metrics_itemk.true_item_dict:
            true_rank = true_item_order.index(item) + 1
            gain = len(true_item_order) - (true_rank - 1)
            ncr_hits.append((item[0], true_rank, gain))
    covered_mass = sum(metrics_itemk.true_item_dict[item] for item in (set(true_item_order) & set(pred_item_order)))

    return RunResult(
        dataname=args.dataname,
        topk=int(case["topk"]),
        epsilon=float(case["epsilon"]),
        repeat_id=int(repeat_id),
        seed=int(seed),
        search_mode=str(case["search_mode"]),
        ratio_group1=float(case["ratios"][0]),
        ratio_group2=float(case["ratios"][1]),
        ratio_group3=float(case["ratios"][2]),
        max_split_order=int(case["max_split_order"]),
        significant_bits_by_depth=case["significant_bits_by_depth"],
        user_ratio_by_depth=case["user_ratio_by_depth"],
        conditional_candidate_k_by_depth=case["conditional_candidate_k_by_depth"],
        global_candidate_k=int(case["global_candidate_k"]),
        effective_root_candidate_k=int(effective_root_candidate_k),
        stage2_topd_ratio=float(case["stage2_topd_ratio"]),
        f1=float(f1),
        ncr=float(ncr),
        mcr=float(mcr),
    )


def summarize_results(results):
    def mean_std_text(values):
        arr = np.array(list(values), dtype=float)
        mean = float(np.mean(arr)) if arr.size > 0 else 0.0
        std = float(np.std(arr)) if arr.size > 0 else 0.0
        return mean, f"{mean:.6f}+-{std:.6f}"

    grouped = {}
    for r in results:
        key = (
            r.topk,
            r.epsilon,
            r.search_mode,
            r.ratio_group1,
            r.ratio_group2,
            r.ratio_group3,
            r.max_split_order,
            r.significant_bits_by_depth,
            r.user_ratio_by_depth,
            r.conditional_candidate_k_by_depth,
            r.global_candidate_k,
            r.effective_root_candidate_k,
            r.stage2_topd_ratio,
        )
        grouped.setdefault(key, []).append(r)

    summary_rows = []
    for key, group in grouped.items():
        f1_mean, f1_text = mean_std_text([x.f1 for x in group])
        ncr_mean, ncr_text = mean_std_text([x.ncr for x in group])
        mcr_mean, mcr_text = mean_std_text([x.mcr for x in group])
        summary_rows.append({
            "topk": key[0],
            "epsilon": key[1],
            "search_mode": key[2],
            "ratio_group1": key[3],
            "ratio_group2": key[4],
            "ratio_group3": key[5],
            "max_split_order": key[6],
            "significant_bits_by_depth": key[7],
            "user_ratio_by_depth": key[8],
            "conditional_candidate_k_by_depth": key[9],
            "global_candidate_k": key[10],
            "effective_root_candidate_k": key[11],
            "stage2_topd_ratio": key[12],
            "repeats": len(group),
            "f1_mean": round(float(f1_mean), 6),
            "ncr_mean": round(float(ncr_mean), 6),
            "mcr_mean": round(float(mcr_mean), 6),
            "f1": f1_text,
            "ncr": ncr_text,
            "mcr": mcr_text,
        })

    summary_rows.sort(
        key=lambda x: (-x["topk"], -x["f1_mean"], -x["ncr_mean"], -x["mcr_mean"])
    )
    return summary_rows


def apply_dataset_defaults(args):
    dataset_key = normalize_dataset_name(args.dataname)
    if dataset_key not in DATASET_RUN_CONFIGS:
        known = ", ".join(sorted(DATASET_RUN_CONFIGS.keys()))
        raise ValueError(f"Unsupported dataname '{args.dataname}'. Supported datasets: {known}")

    cfg = DATASET_RUN_CONFIGS[dataset_key]
    args.dataname = dataset_key
    if args.b is None:
        args.b = int(cfg["b"])
    if args.iterations is None:
        args.iterations = int(cfg["iterations"])
    if args.epsilon is None:
        args.epsilon = list(cfg["case"]["epsilon"])
    return args


def main():
    args = apply_dataset_defaults(parse_args())
    args.epsilon = unique_preserve_order(args.epsilon)
    module = load_target_module()
    cases = expand_cases(curated_cases(args.dataname), args.epsilon)
    cases = sorted(
        cases,
        key=lambda x: (
            -x["topk"],
            x["epsilon"],
            x["search_mode"],
            x["ratios"],
            x["max_split_order"],
            x["global_candidate_k"],
            x["global_only_candidate_k"],
            x["conditional_candidate_k_by_depth"],
            x["stage2_topd_ratio"],
        ),
    )

    dataset_tag = dataset_output_tag(args.dataname)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        THIS_DIR / "results" / f"{dataset_tag}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log_txt = output_dir / f"{dataset_tag}_console.log"
    total_runs = len(cases) * args.repeats
    print(f"total_runs: {total_runs}")

    run_results = []
    run_idx = 0
    with log_txt.open("w", encoding="utf-8") as log_file:
        emit(f"total_runs: {total_runs}", log_file)
        emit(f"runner_module: {TARGET_MODULE.name}", log_file)
        emit(f"b: {args.b}, iterations: {args.iterations}", log_file)
        for case in cases:
            for repeat_id in range(args.repeats):
                run_idx += 1
                emit(
                    f"[{run_idx}/{total_runs}] topk={case['topk']}, epsilon={case['epsilon']}, mode={case['search_mode']}, ratios={case['ratios']}, "
                    f"order={case['max_split_order']}, bits={case['significant_bits_by_depth']}, "
                    f"global_k={case['global_candidate_k']}, global_only_k={case['global_only_candidate_k']}, "
                    f"user_ratio={case['user_ratio_by_depth']}, ck_by_depth={case['conditional_candidate_k_by_depth']}, "
                    f"stage2_topd_ratio={case['stage2_topd_ratio']}, repeat={repeat_id}",
                    log_file,
                )
                result = run_one(module, args, case, repeat_id, log_file)
                run_results.append(result)
                emit(
                    f"[result] f1={result.f1:.6f}, ncr={result.ncr:.6f}, mcr={result.mcr:.6f}\n",
                    log_file,
                )

    runs_jsonl = output_dir / f"{dataset_tag}_runs.jsonl"
    with runs_jsonl.open("w", encoding="utf-8") as f:
        for result in run_results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    runs_csv = output_dir / f"{dataset_tag}_runs.csv"
    with runs_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(run_results[0]).keys()))
        writer.writeheader()
        for result in run_results:
            writer.writerow(asdict(result))

    summary_rows = summarize_results(run_results)
    summary_csv = output_dir / f"{dataset_tag}_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    meta = {
        "dataname": args.dataname,
        "limit": args.limit,
        "epsilon": args.epsilon,
        "repeats": args.repeats,
        "cases": cases,
        "b": args.b,
        "iterations": args.iterations,
        "theta": args.theta,
        "fixed_bit_one_ratio": args.fixed_bit_one_ratio,
        "soft_margin_log": args.soft_margin_log,
        "soft_search_factor": args.soft_search_factor,
        "output_dir": str(output_dir),
    }
    meta_json = output_dir / f"{dataset_tag}_meta.json"
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"saved_console_log: {log_txt}")
    print(f"saved_runs_jsonl: {runs_jsonl}")
    print(f"saved_runs_csv: {runs_csv}")
    print(f"saved_summary_csv: {summary_csv}")
    print(f"saved_meta_json: {meta_json}")


if __name__ == "__main__":
    main()
