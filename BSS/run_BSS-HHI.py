import argparse
import contextlib
import csv
import importlib.util
import io
import math
import random
import sys
from dataclasses import dataclass
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
            "epsilon": [1, 1.5, 2, 2.5, 3, 3.5, 4],
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
        "epsilon": [1, 1.5, 2, 2.5, 3, 3.5, 4],
        "cases": [
            {
                "topk": [32],
                "epsilon": [1, 1.5, 2, 2.5],
                "epsilon_max": 2.5,
                "ratios": [0.1, 0.1, 0.8],
                "search_mode": "full",
                "max_split_order": 1,
                "significant_bits_by_depth": "5",
                "user_ratio_by_depth": "0.5",
                "conditional_candidate_k_by_depth": "50,35,25,15,10",
                "global_candidate_k": 4000,
                "stage2_topd_ratio": [0.3],
            },
            {
                "topk": [32],
                "epsilon": [3, 3.5, 4],
                "epsilon_gt": 2.5,
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
        ],
    },
    "userbehavior1300": {
        "b": 20,
        "iterations": 5,
        "case": {
            "topk": [16, 32],
            "epsilon": [1, 1.5, 2, 2.5, 3, 3.5, 4],
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
            "epsilon": [1, 1.5, 2, 2.5, 3, 3.5, 4],
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
            "epsilon": [1, 1.5, 2, 2.5, 3, 3.5, 4],
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
    epsilon_min=None,
    epsilon_max=None,
    epsilon_lt=None,
    epsilon_gt=None,
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
        "epsilon_min": epsilon_min,
        "epsilon_max": epsilon_max,
        "epsilon_lt": epsilon_lt,
        "epsilon_gt": epsilon_gt,
    }


def curated_cases(dataname: str):
    dataset_key = normalize_dataset_name(dataname)
    if dataset_key not in DATASET_RUN_CONFIGS:
        known = ", ".join(sorted(DATASET_RUN_CONFIGS.keys()))
        raise ValueError(f"Unsupported dataname '{dataname}'. Supported datasets: {known}")
    config = DATASET_RUN_CONFIGS[dataset_key]
    case_configs = config["cases"] if "cases" in config else [config["case"]]
    return [make_case(**case_config) for case_config in case_configs]


def parse_args():
    parser = argparse.ArgumentParser(description="Run curated full BSS-HHI experiments.")
    parser.add_argument("--dataname", type=str, default="kosarak")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--epsilon", type=float, nargs="+", default=None)
    parser.add_argument("--topk", type=int, nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--base-seed", dest="base_seed", type=int, default=20260315)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--dataset-dir", dest="dataset_dir", type=str, default=None)
    parser.add_argument("--middleware-dir", dest="middleware_dir", type=str, default=None)
    return parser.parse_args()


@dataclass
class RunResult:
    dataname: str
    epsilon: float
    topk: int
    f1: float
    ncr: float
    mcr: float


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


def apply_case_overrides(cases, epsilons, topk_values=None):
    overridden = []
    requested_epsilons = [float(epsilon) for epsilon in epsilons]
    matched_epsilons = set()
    for case_config in cases:
        case = dict(case_config)
        epsilon_min = case.get("epsilon_min")
        epsilon_max = case.get("epsilon_max")
        epsilon_lt = case.get("epsilon_lt")
        epsilon_gt = case.get("epsilon_gt")
        eligible_epsilons = [
            epsilon
            for epsilon in requested_epsilons
            if (epsilon_min is None or epsilon >= epsilon_min)
            and (epsilon_max is None or epsilon <= epsilon_max)
            and (epsilon_lt is None or epsilon < epsilon_lt)
            and (epsilon_gt is None or epsilon > epsilon_gt)
        ]
        if not eligible_epsilons:
            continue
        matched_epsilons.update(eligible_epsilons)
        case["epsilon"] = eligible_epsilons
        if topk_values is not None:
            case["topk"] = list(topk_values)
        overridden.append(case)
    unmatched = [epsilon for epsilon in requested_epsilons if epsilon not in matched_epsilons]
    if unmatched:
        raise ValueError(
            "No configured parameter regime for epsilon value(s): "
            + ", ".join(str(value) for value in unmatched)
        )
    return overridden


def build_stage1_union_with_global_scores(bss, stage1_details, b, search_mode):
    if search_mode == "global_only":
        stage1_union = unique_preserve_order(stage1_details["global_decimal_candidates"])
    else:
        stage1_union = unique_preserve_order(bss.collect_stage1_candidate_union(stage1_details))
    global_bit_freq = np.array(stage1_details["global_bit_freq"], dtype=float)
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


def ranked_real_topk(module, data, topk):
    metrics_itemk = module.MetricsItem(data, top_k=topk)
    real_topk_item = metrics_itemk.true_item_dict
    real_topk_ranked = sorted(
        [(item_tuple[0], freq) for item_tuple, freq in real_topk_item.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    return metrics_itemk, real_topk_ranked, {item for item, _ in real_topk_ranked}


def run_one(module, args, case, repeat_id):
    seed = args.base_seed + repeat_id
    random.seed(seed)
    np.random.seed(seed)

    domain_size, user_total = module.resolve_dataset_meta(args.dataname, None, None)
    data = module.Data(
        dataname=args.dataname,
        limit=args.limit,
        domain_size=domain_size,
        user_total=user_total,
        dataset_dir=args.dataset_dir,
        middleware_dir=args.middleware_dir,
    )
    bss = module.BSSHHI(data)
    metrics_itemk, _, _ = ranked_real_topk(module, data, case["topk"])

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

    matrix_side = max(1, int(np.ceil(np.sqrt(max(effective_root_candidate_k, 1)))))
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, stage1_details = bss.init_matrix(
            group1=group1,
            epsilon=case["epsilon"],
            topk=matrix_side,
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
        )

    stage1_item_ids, stage1_item_scores = build_stage1_union_with_global_scores(
        bss, stage1_details, args.b, case["search_mode"]
    )
    stage1_union = list(stage1_item_ids)
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
    group3, user_num = bss.remove_non_top_items(
        group3,
        stage2_candidates,
    )

    svim = module.SVIM(group3, top_k=case["topk"], epsilon=case["epsilon"])
    with contextlib.redirect_stdout(io.StringIO()):
        estimate_list_svim, estimate_freq_svim = svim.find_singleton_origin(user_num)

    estimate_list_svim = bss.recover_items_from_top_list(estimate_list_svim, stage2_candidates)
    estimate_dict_svim = svim.convert_result2dict(estimate_list_svim, estimate_freq_svim)

    f1 = metrics_itemk.item_f1(estimate_dict_svim)
    ncr = metrics_itemk.item_ncr(estimate_dict_svim)
    mcr = metrics_itemk.cal_mcr(estimate_dict_svim)

    return RunResult(
        dataname=args.dataname,
        epsilon=float(case["epsilon"]),
        topk=int(case["topk"]),
        f1=float(f1),
        ncr=float(ncr),
        mcr=float(mcr),
    )


def summarize_results(results):
    def mean_std_text(values):
        arr = np.array(list(values), dtype=float)
        mean = float(np.mean(arr)) if arr.size > 0 else 0.0
        std = float(np.std(arr)) if arr.size > 0 else 0.0
        return f"{mean:.6f}+-{std:.6f}"

    grouped = {}
    for r in results:
        key = (r.dataname, r.epsilon, r.topk)
        grouped.setdefault(key, []).append(r)

    summary_rows = []
    for key, group in grouped.items():
        summary_rows.append({
            "dataset": key[0],
            "epsilon": key[1],
            "k": key[2],
            "f1": mean_std_text([x.f1 for x in group]),
            "ncr": mean_std_text([x.ncr for x in group]),
            "mcr": mean_std_text([x.mcr for x in group]),
        })

    summary_rows.sort(key=lambda x: (x["dataset"], x["epsilon"], x["k"]))
    return summary_rows


def apply_dataset_defaults(args):
    dataset_key = normalize_dataset_name(args.dataname)
    if dataset_key not in DATASET_RUN_CONFIGS:
        known = ", ".join(sorted(DATASET_RUN_CONFIGS.keys()))
        raise ValueError(f"Unsupported dataname '{args.dataname}'. Supported datasets: {known}")

    cfg = DATASET_RUN_CONFIGS[dataset_key]
    args.dataname = dataset_key
    args.b = int(cfg["b"])
    args.iterations = int(cfg["iterations"])
    args.theta = 0.9
    args.global_user_ratio = 0.5
    args.node_user_ratio = 0.5
    args.fixed_bit_one_ratio = 0.45
    args.soft_margin_log = 0.05
    args.soft_search_factor = 2
    if args.epsilon is None:
        if "epsilon" in cfg:
            args.epsilon = list(cfg["epsilon"])
        else:
            args.epsilon = list(cfg["case"]["epsilon"])
    return args


def main():
    args = apply_dataset_defaults(parse_args())
    args.epsilon = unique_preserve_order(args.epsilon)
    module = load_target_module()
    base_cases = apply_case_overrides(
        curated_cases(args.dataname),
        args.epsilon,
        args.topk,
    )
    cases = expand_cases(base_cases, args.epsilon)
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

    run_results = []
    for case in cases:
        for repeat_id in range(args.repeats):
            run_results.append(run_one(module, args, case, repeat_id))

    results_csv = output_dir / "results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["dataset", "epsilon", "k", "run", "f1", "ncr", "mcr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        run_counts = {}
        for result in run_results:
            key = (result.dataname, result.epsilon, result.topk)
            run_counts[key] = run_counts.get(key, 0) + 1
            writer.writerow({
                "dataset": result.dataname,
                "epsilon": result.epsilon,
                "k": result.topk,
                "run": run_counts[key],
                "f1": f"{result.f1:.6f}",
                "ncr": f"{result.ncr:.6f}",
                "mcr": f"{result.mcr:.6f}",
            })
        for row in summarize_results(run_results):
            writer.writerow({**row, "run": "mean+-std"})



if __name__ == "__main__":
    main()
