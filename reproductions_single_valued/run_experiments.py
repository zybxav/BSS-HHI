"""Run the reviewer-facing HHI and frequency-oracle reproductions."""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from reproductions.common.dataset import load_single_values
from reproductions.common.io import NumpyEncoder, save_csv
from reproductions.common.metrics import fo_metrics, hhi_metrics, top_k_items
from reproductions.registry import FO_METHODS, HHI_METHODS


DEFAULT_EPSILONS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
DEFAULT_SEEDS = (101, 102, 103, 104, 105)


def run_hhi(method, values, domain_size, epsilon, top_k, seed, truth_one):
    result = HHI_METHODS[method]().run(values, domain_size, epsilon, top_k, seed)
    items = list(map(int, result["items"]))
    record = hhi_metrics(truth_one, items, top_k)
    record.update(
        {
            "items": items,
            "estimated_frequencies": result["estimated_frequencies"],
            "candidate_count": int(result["candidate_count"]),
            "runtime_client": float(result["runtime_client"]),
            "runtime_server": float(result["runtime_server"]),
            "communication_bits_per_user": result["communication_bits_per_user"],
            "diagnostics": result.get("diagnostics", {}),
        }
    )
    return record


def run_fo(method, values, domain_size, epsilon, top_k, seed, truth_one):
    queries = np.arange(domain_size, dtype=np.int64)
    oracle = FO_METHODS[method]()
    estimates = np.asarray(
        oracle.estimate(values, domain_size, epsilon, seed, queries), dtype=float
    )
    order = np.lexsort((queries, -np.nan_to_num(estimates, nan=-np.inf)))[:top_k]
    items = (queries[order] + 1).tolist()
    truth_counts = np.bincount(values, minlength=domain_size).astype(float)
    errors = fo_metrics(truth_counts, estimates)
    record = hhi_metrics(truth_one, items, top_k)
    record.update(
        {
            "items": items,
            "estimated_frequencies": estimates[order].tolist(),
            "candidate_count": int(domain_size),
            "count_mse": errors["mse"],
            "frequency_mse": errors["mse"] / (values.size**2),
            "max_abs_frequency_error": errors["max_abs_error"] / values.size,
            "diagnostics": oracle.parameters(),
        }
    )
    record["runtime_client"] = record["diagnostics"].get("runtime_client")
    record["runtime_server"] = record["diagnostics"].get("runtime_server")
    record["communication_bits_per_user"] = record["diagnostics"].get(
        "communication_bits_per_user"
    )
    return record


def csv_row(record):
    fields = (
        "dataset",
        "family",
        "method",
        "epsilon",
        "seed",
        "users",
        "domain_size",
        "top_k",
        "hits",
        "precision",
        "recall",
        "f1",
        "count_mse",
        "frequency_mse",
        "max_abs_frequency_error",
        "candidate_count",
        "runtime_client",
        "runtime_server",
        "wall_time",
        "communication_bits_per_user",
        "status",
    )
    return {field: record.get(field) for field in fields}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("prepared_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("run_output"))
    parser.add_argument("--families", nargs="+", choices=("HHI", "FO"), default=("HHI", "FO"))
    parser.add_argument("--hhi-methods", nargs="+", choices=tuple(HHI_METHODS), default=tuple(HHI_METHODS))
    parser.add_argument("--fo-methods", nargs="+", choices=tuple(FO_METHODS), default=tuple(FO_METHODS))
    parser.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in manifest["datasets"]:
        data_path = args.data_dir / dataset["output"]
        values = load_single_values(str(data_path))
        domain_size = int(dataset["domain_size"])
        truth_one = [x + 1 for x in top_k_items(values, args.top_k)]
        for family in args.families:
            methods = args.hhi_methods if family == "HHI" else args.fo_methods
            for method in methods:
                for epsilon in args.epsilons:
                    for seed in args.seeds:
                        record = {
                            "dataset": dataset["dataset"],
                            "dataset_sha256": dataset["output_sha256"],
                            "family": family,
                            "method": method,
                            "epsilon": float(epsilon),
                            "seed": int(seed),
                            "users": int(values.size),
                            "domain_size": domain_size,
                            "top_k": int(args.top_k),
                            "truth_items": truth_one,
                        }
                        started = time.perf_counter()
                        try:
                            if family == "HHI":
                                record.update(run_hhi(method, values, domain_size, epsilon, args.top_k, seed, truth_one))
                            else:
                                record.update(run_fo(method, values, domain_size, epsilon, args.top_k, seed, truth_one))
                            record["status"] = "ok"
                        except Exception as exc:
                            record.update(
                                {
                                    "status": f"error: {type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc(),
                                }
                            )
                            if args.fail_fast:
                                raise
                        record["wall_time"] = time.perf_counter() - started
                        records.append(record)
                        print(
                            f"{record['dataset']} {family} {method} "
                            f"eps={epsilon:g} seed={seed} {record['status']}"
                        )

    with (args.output_dir / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, cls=NumpyEncoder) + "\n")
    save_csv(str(args.output_dir / "runs.csv"), [csv_row(x) for x in records])
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "data_manifest": str(manifest_path.resolve()),
                "families": list(args.families),
                "hhi_methods": list(args.hhi_methods),
                "fo_methods": list(args.fo_methods),
                "epsilons": list(args.epsilons),
                "seeds": list(args.seeds),
                "top_k": int(args.top_k),
                "records": len(records),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
