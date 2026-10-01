"""Merge the raw rows of several runs, so their reports can be rebuilt as one data set."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_latency_bench.results import Row, read_raw_csv


class CombineError(ValueError):
    pass


def tokens_label(tokens: int) -> str:
    """A size label that states its size, so runs with different label names can share one table."""
    return f"{tokens / 1000:g}k"


def _size_targets(metadata: dict[str, Any], run_dir: Path) -> tuple[dict[str, int], dict[str, int]]:
    try:
        sizes = metadata["settings"]["sizes"]
        return dict(sizes["prompt_tokens"]), dict(sizes["context_tokens"])
    except (KeyError, TypeError) as error:
        raise CombineError(f"{run_dir}: run_metadata.json has no size settings") from error


def _target(targets: dict[str, int], label: str, run_dir: Path) -> int:
    if label not in targets:
        raise CombineError(f"{run_dir}: size {label!r} in raw.csv is not in run_metadata.json")
    return targets[label]


def _load_run(run_dir: Path) -> tuple[list[Row], dict[str, Any]]:
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    prompt_targets, context_targets = _size_targets(metadata, run_dir)
    rows = read_raw_csv(run_dir / "raw.csv")
    for row in rows:
        if row["phase"] != "measure":
            continue
        # Older runs lack target-token columns, so take them from the run's settings.
        if row["prompt_target_tokens"] is None:
            row["prompt_target_tokens"] = _target(prompt_targets, row["prompt_size"], run_dir)
        if row["context_target_tokens"] is None:
            row["context_target_tokens"] = _target(context_targets, row["context_size"], run_dir)
        row["context_size"] = tokens_label(row["context_target_tokens"])
    return rows, metadata["settings"].get("output", {})


def combine_runs(run_dirs: Sequence[Path]) -> list[Row]:
    """All rows of the given runs. Context sizes are relabelled by their target tokens, for example `16k`.

    Prompt labels are kept, so a label must mean the same target size in every run. The runs must share their
    output settings, because the answer length and token cap change what is measured.
    """
    rows: list[Row] = []
    prompt_targets: dict[str, int] = {}
    seen_runs: set[str] = set()
    first_output: dict[str, Any] | None = None
    for run_dir in run_dirs:
        run_rows, output = _load_run(run_dir)
        if first_output is None:
            first_output = output
        elif output != first_output:
            raise CombineError(f"{run_dir}: output settings {output} differ from {first_output} in the first run")
        run_ids = {row["run_id"] for row in run_rows}
        if run_ids & seen_runs:
            raise CombineError(f"{run_dir}: run {', '.join(sorted(run_ids & seen_runs))} is listed twice")
        seen_runs |= run_ids
        for row in run_rows:
            if row["phase"] != "measure":
                continue
            label, tokens = row["prompt_size"], row["prompt_target_tokens"]
            if prompt_targets.setdefault(label, tokens) != tokens:
                raise CombineError(
                    f"{run_dir}: prompt size {label!r} is {tokens} tokens here but {prompt_targets[label]} elsewhere"
                )
        rows.extend(run_rows)
    return rows
