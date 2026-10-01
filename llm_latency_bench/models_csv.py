"""Load and validate the model/reasoning matrix from `models.csv`."""

import csv
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("model_name", "reasoning_level_name", "model_slug", "reasoning_level")


class ModelsCsvError(ValueError):
    pass


@dataclass(frozen=True)
class ModelPair:
    model_name: str
    reasoning_level_name: str
    model_slug: str
    reasoning_level: str

    @property
    def key(self) -> str:
        return f"{self.model_slug}/{self.reasoning_level}"


def load_model_pairs(path: Path) -> list[ModelPair]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ModelsCsvError(f"{path}: missing required column(s): {', '.join(missing)}")

        pairs: list[ModelPair] = []
        seen: dict[tuple[str, str], int] = {}
        # Line 1 is the header.
        for line_number, row in enumerate(reader, start=2):
            values = {column: (row.get(column) or "").strip() for column in REQUIRED_COLUMNS}
            blank = [column for column, value in values.items() if not value]
            if blank:
                raise ModelsCsvError(f"{path}:{line_number}: blank value(s) in column(s): {', '.join(blank)}")
            pair = ModelPair(**values)
            identity = (pair.model_slug, pair.reasoning_level)
            if identity in seen:
                raise ModelsCsvError(
                    f"{path}:{line_number}: duplicate model/reasoning pair {pair.key} "
                    f"(first seen on line {seen[identity]})"
                )
            seen[identity] = line_number
            pairs.append(pair)

    if not pairs:
        raise ModelsCsvError(f"{path}: no model rows")
    return pairs


def filter_pairs(
    pairs: list[ModelPair], models: Collection[str] = (), reasoning_levels: Collection[str] = ()
) -> list[ModelPair]:
    """Keep pairs whose slug and reasoning level match the filters. An empty filter keeps everything."""
    unknown_models = set(models) - {pair.model_slug for pair in pairs}
    unknown_levels = set(reasoning_levels) - {pair.reasoning_level for pair in pairs}
    if unknown_models or unknown_levels:
        unknown = sorted(unknown_models | unknown_levels)
        raise ModelsCsvError(f"filter value(s) not present in models.csv: {', '.join(unknown)}")
    selected = [
        pair
        for pair in pairs
        if (not models or pair.model_slug in models)
        and (not reasoning_levels or pair.reasoning_level in reasoning_levels)
    ]
    if not selected:
        raise ModelsCsvError("filters matched no model/reasoning pair")
    return selected
