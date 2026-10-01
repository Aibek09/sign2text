#!/usr/bin/env python3
"""Evaluate Sign2Text predictions from JSONL without third-party packages."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def edit_distance(reference: Sequence[str], prediction: Sequence[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, pred_item in enumerate(prediction, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref_item != pred_item)))
        previous = current
    return previous[-1]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def score(records: list[dict]) -> dict:
    if not records:
        raise ValueError("No predictions to evaluate")
    word_edits = word_total = char_edits = char_total = exact = 0
    latencies: list[float] = []
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if not isinstance(record.get("reference"), str) or not normalize(record["reference"]):
            raise ValueError("Each record requires a nonempty English reference")
        if not isinstance(record.get("prediction", ""), str):
            raise ValueError("Prediction must be text; use an empty string for no prediction")
        ref = normalize(record["reference"])
        pred = normalize(record.get("prediction", ""))
        ref_words, pred_words = ref.split(), pred.split()
        word_edits += edit_distance(ref_words, pred_words)
        word_total += len(ref_words)
        char_edits += edit_distance(list(ref.replace(" ", "")), list(pred.replace(" ", "")))
        char_total += len(ref.replace(" ", ""))
        exact += ref == pred
        if record.get("latency_ms") is not None:
            latency = float(record["latency_ms"])
            if not math.isfinite(latency) or latency < 0:
                raise ValueError("Latency must be finite and nonnegative")
            latencies.append(latency)
        by_condition[record.get("condition", "unspecified")].append(record)

    result = {
        "samples": len(records),
        "wer": word_edits / max(1, word_total),
        "cer": char_edits / max(1, char_total),
        "exact_match": exact / max(1, len(records)),
        "latency_ms": {"mean": statistics.fmean(latencies) if latencies else None, "p50": percentile(latencies, .5), "p95": percentile(latencies, .95)},
    }
    slices = {}
    for condition, subset in by_condition.items():
        if len(by_condition) == 1 and condition == "unspecified":
            continue
        sliced = score([{**item, "condition": "unspecified"} for item in subset])
        slices[condition] = {"samples": sliced["samples"], "wer": sliced["wer"], "cer": sliced["cer"], "exact_match": sliced["exact_match"]}
    if slices:
        clean_wer = slices.get("clean", {}).get("wer")
        for values in slices.values():
            values["wer_gap_vs_clean"] = values["wer"] - clean_wer if clean_wer is not None else None
        result["conditions"] = slices
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.input.resolve() == args.output.resolve():
        parser.error("Input and output paths must differ; predictions will not be overwritten")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = score(records)
    modes = {r.get("mode", "unspecified") for r in records}
    if len(modes) > 1:
        result["by_mode"] = {mode: score([r for r in records if r.get("mode", "unspecified") == mode]) for mode in sorted(modes)}
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
