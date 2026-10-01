"""Non-secret benchmark settings loaded from `settings.toml`."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_type_hints

from llm_latency_bench.fixtures import MAX_TURNS


@dataclass(frozen=True)
class ClientSettings:
    connect_timeout_s: float
    read_timeout_s: float
    request_timeout_s: float
    keepalive_expiry_s: float


@dataclass(frozen=True)
class RetrySettings:
    max_retries: int
    backoff_base_s: float
    backoff_max_s: float


@dataclass(frozen=True)
class RunSettings:
    repetitions: int
    seed: int
    concurrency: int
    warmup: bool


@dataclass(frozen=True)
class OutputSettings:
    answer_words: int
    max_tokens: int


@dataclass(frozen=True)
class SizeSettings:
    turn_counts: tuple[int, ...]
    # Size label -> approximate target tokens, in the order given in the settings file.
    prompt_tokens: dict[str, int]
    context_tokens: dict[str, int]


@dataclass(frozen=True)
class StatsSettings:
    min_samples_p95: int
    min_samples_p99: int


@dataclass(frozen=True)
class Settings:
    client: ClientSettings
    retries: RetrySettings
    run: RunSettings
    output: OutputSettings
    sizes: SizeSettings
    stats: StatsSettings


class SettingsError(ValueError):
    pass


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise SettingsError(f"settings: missing [{name}] table")
    return section


def _build[T](cls: type[T], data: dict[str, Any], name: str) -> T:
    try:
        built = cls(**data)
    except TypeError as error:
        raise SettingsError(f"settings: invalid [{name}] table: {error}") from error
    for field_name, expected in get_type_hints(cls).items():
        value = data[field_name]
        # A float setting may be written as `10`, and bool is a subclass of int.
        allowed = (int, float) if expected is float else (expected,)
        if not isinstance(value, allowed) or (isinstance(value, bool) and expected is not bool):
            raise SettingsError(f"settings: {name}.{field_name} must be a {expected.__name__}")
    return built


def _size_levels(data: object, name: str) -> dict[str, int]:
    if not isinstance(data, dict) or not data:
        raise SettingsError(f"settings: [{name}] must be a non-empty table of label = tokens")
    if any(isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1 for tokens in data.values()):
        raise SettingsError(f"settings: every value in [{name}] must be a positive integer")
    return dict(data)


def load_settings(path: Path) -> Settings:
    with path.open("rb") as handle:
        try:
            data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as error:
            raise SettingsError(f"settings: {path} is not valid TOML: {error}") from error

    sizes = dict(_section(data, "sizes"))
    turn_counts = tuple(sizes.pop("turn_counts", ()))
    prompt_tokens = _size_levels(sizes.pop("prompt_tokens", None), "sizes.prompt_tokens")
    context_tokens = _size_levels(sizes.pop("context_tokens", None), "sizes.context_tokens")
    if sizes:
        raise SettingsError(f"settings: unknown key(s) in [sizes]: {', '.join(sizes)}")
    if (
        not turn_counts
        or len(set(turn_counts)) != len(turn_counts)
        or any(
            isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_TURNS
            for count in turn_counts
        )
    ):
        raise SettingsError(
            f"settings: sizes.turn_counts must be a non-empty list of distinct integers from 1 to {MAX_TURNS}"
        )

    settings = Settings(
        client=_build(ClientSettings, _section(data, "client"), "client"),
        retries=_build(RetrySettings, _section(data, "retries"), "retries"),
        run=_build(RunSettings, _section(data, "run"), "run"),
        output=_build(OutputSettings, _section(data, "output"), "output"),
        sizes=SizeSettings(turn_counts=turn_counts, prompt_tokens=prompt_tokens, context_tokens=context_tokens),
        stats=_build(StatsSettings, _section(data, "stats"), "stats"),
    )
    if settings.run.concurrency < 1 or settings.run.repetitions < 1:
        raise SettingsError("settings: run.concurrency and run.repetitions must be at least 1")
    if settings.retries.max_retries < 0:
        raise SettingsError("settings: retries.max_retries must not be negative")
    return settings
