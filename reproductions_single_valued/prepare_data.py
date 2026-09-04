"""Convert set-valued user rows to the shared 1-based single-valued input.

Each non-empty line is one user and contains comma- or whitespace-separated,
positive item IDs. One distinct item is sampled uniformly per user. An optional
second step samples users uniformly without replacement.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_dataset(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError("use NAME=PATH")
    name, raw_path = spec.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("use NAME=PATH")
    return name.strip(), Path(raw_path).expanduser()


def convert(source, conversion_seed, sample_users, sample_seed):
    rng = np.random.default_rng(int(conversion_seed))
    selected = []
    source_rows = 0
    source_tokens = 0
    domain_size = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            source_rows += 1
            try:
                items = [int(x) for x in raw.replace(",", " ").split()]
            except ValueError as exc:
                raise ValueError(f"{source}: non-integer token at line {line_no}") from exc
            if not items:
                continue
            if min(items) < 1:
                raise ValueError(f"{source}: IDs must be positive and 1-based")
            source_tokens += len(items)
            domain_size = max(domain_size, max(items))
            unique_items = np.unique(np.asarray(items, dtype=np.int64))
            selected.append(int(unique_items[rng.integers(unique_items.size)]))
    if not selected:
        raise ValueError(f"{source}: no non-empty users")
    values = np.asarray(selected, dtype=np.int64)
    if sample_users is not None and values.size > sample_users:
        sample_rng = np.random.default_rng(int(sample_seed))
        indices = sample_rng.choice(values.size, size=int(sample_users), replace=False)
        values = values[indices]
    return values, {
        "source_rows": source_rows,
        "nonempty_users": len(selected),
        "source_tokens": source_tokens,
        "domain_size": domain_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_dataset,
        metavar="NAME=PATH",
        help="repeat for each dataset",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("prepared_data"))
    parser.add_argument("--conversion-seed", type=int, default=20260724)
    parser.add_argument("--sample-users", type=int, default=100000)
    parser.add_argument("--sample-seed", type=int, default=20260725)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, source in args.input:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = args.output_dir / f"{name}.single.txt"
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"{target} exists; pass --overwrite")
        values, stats = convert(
            source,
            args.conversion_seed,
            args.sample_users,
            args.sample_seed,
        )
        np.savetxt(target, values, fmt="%d")
        counts = Counter(map(int, values))
        top32 = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:32]
        records.append(
            {
                "dataset": name,
                "source": str(source),
                "source_sha256": file_hash(source),
                "output": target.name,
                "output_sha256": file_hash(target),
                "stored_id_base": 1,
                "conversion_seed": int(args.conversion_seed),
                "sample_seed": int(args.sample_seed),
                "users_requested": int(args.sample_users),
                "users": int(values.size),
                "domain_size": int(stats["domain_size"]),
                "source_rows": int(stats["source_rows"]),
                "nonempty_users": int(stats["nonempty_users"]),
                "source_tokens": int(stats["source_tokens"]),
                "top32": [
                    {"item": item, "count": count, "frequency": count / values.size}
                    for item, count in top32
                ],
            }
        )
        print(f"{name}: users={values.size}, domain={stats['domain_size']}")

    manifest = {
        "format": "one positive 1-based item ID per line",
        "sampling": "one uniformly sampled distinct item per non-empty user row",
        "datasets": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
