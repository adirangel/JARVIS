"""CLI for collecting and normalizing transcripts into JSONL records."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from .dataset_utils import iter_transcript_turns, load_metadata, write_jsonl


def discover_transcript_files(source: Path, pattern: str) -> List[Path]:
    if source.is_file():
        return [source]
    files = sorted(source.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No transcripts matching {pattern!r} under {source}")
    return files


def collect(
    source: Path,
    output: Path,
    *,
    pattern: str = "*.txt",
    source_name: str = "unknown",
    metadata_path: Optional[Path] = None,
    id_prefix: str = "",
) -> None:
    metadata = load_metadata(metadata_path)
    records = []
    for transcript_path in discover_transcript_files(source, pattern):
        for turn in iter_transcript_turns(
            transcript_path,
            source=source_name,
            metadata=metadata,
            id_prefix=id_prefix,
        ):
            records.append(turn.to_dict())
    write_jsonl(output, records)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="File or directory containing transcripts")
    parser.add_argument("output", type=Path, help="Destination JSONL file")
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern for transcript discovery when SOURCE is a directory",
    )
    parser.add_argument("--source-name", default="unknown", help="Source identifier stored in records")
    parser.add_argument("--metadata", type=Path, help="Optional JSON file with conversation metadata")
    parser.add_argument("--id-prefix", default="", help="Prefix prepended to generated turn IDs")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    collect(
        args.source,
        args.output,
        pattern=args.pattern,
        source_name=args.source_name,
        metadata_path=args.metadata,
        id_prefix=args.id_prefix,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
