#!/usr/bin/env python3
"""Create a small, reviewable How2Sign topic manifest from CSV metadata."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_KEYWORDS = {
    "coffee", "tea", "water", "drink", "cup", "milk", "sugar", "size", "small", "medium", "large",
    "order", "want", "would like", "please", "thank", "pay", "cash", "card", "restroom", "bathroom",
    "hot", "cold", "iced", "food", "sandwich", "table", "here", "take out", "how much",
}
TEXT_COLUMNS = ("SENTENCE", "sentence", "text", "TEXT", "translation", "TRANSLATION")
ID_COLUMNS = ("SENTENCE_NAME", "sentence_name", "id", "ID", "clip_id")
VIDEO_COLUMNS = ("VIDEO_NAME", "video_name", "video", "VIDEO", "path", "url")


def first(row: dict[str, str], names: tuple[str, ...]) -> str:
    return next((row[name].strip() for name in names if isinstance(row.get(name), str) and row[name].strip()), "")


def matches_topic(text: str, keywords: set[str]) -> list[str]:
    normalized = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    return sorted(keyword for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", normalized))


def build_manifest(input_path: Path, output_path: Path, keywords: set[str] | None, limit: int | None) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with input_path.open(newline="", encoding="utf-8-sig") as source:
        sample = source.read(8192)
        source.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(source, dialect=dialect)
        if not reader.fieldnames or not any(name in reader.fieldnames for name in TEXT_COLUMNS):
            raise ValueError("Metadata must contain a sentence/text column")
        records = []
        for index, row in enumerate(reader, start=1):
            text = first(row, TEXT_COLUMNS)
            matched = matches_topic(text, keywords) if keywords else []
            if not text or (keywords and not matched):
                continue
            record = {
                "id": first(row, ID_COLUMNS) or f"row-{index}",
                "text": text,
                "video": first(row, VIDEO_COLUMNS),
                "topic": "filtered" if keywords else "all",
                "matched_keywords": matched,
                "review_status": "pending",
                "source_row": index,
                "start": row.get("START", row.get("start", "")),
                "end": row.get("END", row.get("end", "")),
                "signer_id": first(row, ("SIGNER_ID", "signer_id", "signer")),
                "split": first(row, ("SPLIT", "split")),
            }
            records.append(record)
            kept += 1
            if limit and kept >= limit:
                break
    with output_path.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--keywords", help="Comma-separated override")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--topic", choices=["all", "cafe"], default="all")
    args = parser.parse_args()
    keywords = {item.strip().lower() for item in args.keywords.split(",") if item.strip()} if args.keywords else (DEFAULT_KEYWORDS if args.topic == "cafe" else None)
    count = build_manifest(args.input, args.output, keywords, args.limit)
    print(f"Wrote {count} candidate clips to {args.output}")


if __name__ == "__main__":
    main()
