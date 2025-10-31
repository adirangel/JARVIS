"""Aggregate normalized transcripts and produce dataset splits."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .dataset_utils import write_jsonl


class RatioError(ValueError):
    """Raised when split ratios do not sum to one."""


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def deduplicate(records: Sequence[dict], fields: Sequence[str]) -> List[dict]:
    if not fields:
        return list(records)
    seen = set()
    unique: List[dict] = []
    for record in records:
        key = tuple(record.get(field) for field in fields)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def validate_ratios(train: float, val: float, test: float, tol: float = 1e-6) -> None:
    if abs((train + val + test) - 1.0) > tol:
        raise RatioError("Split ratios must sum to 1.0")


def partition(records: Sequence[dict], train: float, val: float, test: float) -> List[List[dict]]:
    total = len(records)
    train_count = int(total * train)
    val_count = int(total * val)
    test_count = total - train_count - val_count
    return [
        list(records[:train_count]),
        list(records[train_count : train_count + val_count]),
        list(records[train_count + val_count : train_count + val_count + test_count]),
    ]


def save_split(records: Sequence[dict], output: Path, fmt: str) -> None:
    if fmt == "jsonl":
        write_jsonl(output, records)
    elif fmt == "parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Pandas with a parquet backend is required for parquet output") from exc
        df = pd.DataFrame(records)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def build_dataset(
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    fmt: str,
    dedupe_fields: Sequence[str],
    seed: Optional[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    validate_ratios(train_ratio, val_ratio, test_ratio)
    all_records: List[dict] = []
    for path in inputs:
        all_records.extend(read_jsonl(path))

    all_records = deduplicate(all_records, dedupe_fields)
    if seed is not None:
        random.Random(seed).shuffle(all_records)
    else:
        random.shuffle(all_records)

    train_records, val_records, test_records = partition(all_records, train_ratio, val_ratio, test_ratio)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_split(train_records, output_dir / f"train.{fmt}", fmt)
    save_split(val_records, output_dir / f"validation.{fmt}", fmt)
    save_split(test_records, output_dir / f"test.{fmt}", fmt)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Normalized JSONL files to combine")
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"), help="Destination directory")
    parser.add_argument("--format", choices=["jsonl", "parquet"], default="jsonl")
    parser.add_argument("--dedupe-on", nargs="*", default=["text"], help="Fields used for deduplication")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    build_dataset(
        args.inputs,
        args.output_dir,
        fmt=args.format,
        dedupe_fields=args.dedupe_on,
        seed=args.seed,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
