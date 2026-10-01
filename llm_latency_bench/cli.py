"""Command line entry point: `llm-latency-bench run`."""

import argparse
import asyncio
import dataclasses
import json
import secrets
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from llm_latency_bench.combine import CombineError, combine_runs
from llm_latency_bench.credentials import CredentialsError, Redactor, load_api_key
from llm_latency_bench.models_csv import ModelPair, ModelsCsvError, filter_pairs, load_model_pairs
from llm_latency_bench.openrouter import build_http_client, model_factory
from llm_latency_bench.report import summarize
from llm_latency_bench.results import AGGREGATED_COLUMNS, RAW_COLUMNS, RawWriter, Row, aggregate, write_csv
from llm_latency_bench.runner import BenchmarkRunner, RunOutcome, run_matrix
from llm_latency_bench.scenarios import Scenario, build_scenarios, estimate_input_tokens, expected_requests
from llm_latency_bench.settings import Settings, SettingsError, load_settings
from llm_latency_bench.tracing import local_tracing


@dataclass(frozen=True)
class RunPlan:
    settings: Settings
    pairs: list[ModelPair]
    scenarios: list[Scenario]
    repetitions: int
    seed: int
    concurrency: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-latency-bench", description="Streaming latency benchmark through OpenRouter."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the selected model/reasoning matrix once and write the result files")
    run.add_argument("--models-csv", type=Path, default=Path("models.csv"), help="model matrix (default: models.csv)")
    run.add_argument("--settings", type=Path, default=Path("settings.toml"), help="default: settings.toml")
    run.add_argument("--env-file", type=Path, default=Path(".env"), help="file with OPENROUTER_API_KEY (default: .env)")
    run.add_argument("--output-dir", type=Path, default=Path("outputs"), help="results go to <dir>/<run id>/")
    run.add_argument("--repetitions", type=int, help="repetitions of every scenario (default: settings.toml)")
    run.add_argument("--seed", type=int, help="seed for the scenario order (default: settings.toml)")
    run.add_argument("--concurrency", type=int, help="scenarios in flight at once (default: settings.toml)")
    run.add_argument("--model", action="append", default=[], metavar="SLUG", help="only this model slug (repeatable)")
    run.add_argument(
        "--reasoning", action="append", default=[], metavar="LEVEL", help="only this reasoning level (repeatable)"
    )
    run.add_argument("--dry-run", action="store_true", help="print the plan and exit without sending requests")

    combine = commands.add_parser("combine", help="merge the raw rows of several runs and rebuild the reports")
    combine.add_argument("run_dirs", nargs="+", type=Path, metavar="RUN_DIR", help="run folders with raw.csv")
    combine.add_argument("--output-dir", type=Path, required=True, help="new folder for the combined files")
    combine.add_argument(
        "--settings",
        type=Path,
        default=Path("settings.toml"),
        help="source of the tail thresholds (default: settings.toml)",
    )
    return parser


def _build_plan(args: argparse.Namespace) -> RunPlan:
    settings = load_settings(args.settings)

    def pick(value: int | None, default: int) -> int:
        return default if value is None else value

    repetitions = pick(args.repetitions, settings.run.repetitions)
    concurrency = pick(args.concurrency, settings.run.concurrency)
    seed = pick(args.seed, settings.run.seed)
    if repetitions < 1 or concurrency < 1:
        raise SettingsError("--repetitions and --concurrency must be at least 1")

    pairs = filter_pairs(load_model_pairs(args.models_csv), args.model, args.reasoning)
    scenarios = build_scenarios(pairs, settings.sizes, repetitions, seed)
    return RunPlan(settings, pairs, scenarios, repetitions, seed, concurrency)


def _plan_text(plan: RunPlan) -> str:
    settings = plan.settings
    warmups = len(plan.pairs) if settings.run.warmup else 0
    requests = expected_requests(plan.scenarios)
    per_pair = Counter(scenario.pair.key for scenario in plan.scenarios)
    tokens = estimate_input_tokens(plan.scenarios, settings.output.answer_words)
    return "\n".join(
        [
            f"pairs ({len(plan.pairs)}): "
            + ", ".join(f"{pair.key} ({per_pair[pair.key]} scenarios)" for pair in plan.pairs),
            f"planned scenarios: {len(plan.scenarios)} ({len(settings.sizes.prompt_tokens)} prompt sizes x "
            f"{len(settings.sizes.context_tokens)} context sizes x {len(settings.sizes.turn_counts)} turn counts x "
            f"{plan.repetitions} repetition(s) per pair)",
            f"expected requests: {requests + warmups} ({requests} scenario turns + {warmups} warm-up), "
            "excluding retries",
            f"estimated input tokens: ~{tokens:,} before prompt-cache discounts (text length / 4)",
        ]
    )


def _metadata(run_id: str, plan: RunPlan, models_csv: Path) -> dict[str, object]:
    return {
        "run_id": run_id,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "seed": plan.seed,
        "repetitions": plan.repetitions,
        "concurrency": plan.concurrency,
        "models_csv": str(models_csv),
        "pairs": [dataclasses.asdict(pair) for pair in plan.pairs],
        "settings": dataclasses.asdict(plan.settings),
        "endpoint": "openrouter chat/completions (streaming)",
        "versions": {name: version(name) for name in ("pydantic-ai-slim", "openai", "logfire", "httpx2")},
    }


async def _benchmark(run_id: str, plan: RunPlan, api_key: str, writer: RawWriter, redact: Redactor) -> RunOutcome:
    async with build_http_client(plan.settings.client) as http_client:
        runner = BenchmarkRunner(
            run_id=run_id,
            seed=plan.seed,
            settings=plan.settings,
            model_factory=model_factory(api_key, http_client),
            writer=writer,
            redact=redact,
        )
        return await run_matrix(
            runner, plan.pairs, plan.scenarios, concurrency=plan.concurrency, warmup=plan.settings.run.warmup
        )


def _files_line(folder: Path) -> str:
    return "files: " + ", ".join(str(folder / name) for name in ("raw.csv", "aggregated.csv", "summary.txt"))


def _write_reports(run_dir: Path, rows: list[Row], settings: Settings) -> str:
    stats = settings.stats
    write_csv(
        run_dir / "aggregated.csv", AGGREGATED_COLUMNS, aggregate(rows, stats.min_samples_p95, stats.min_samples_p99)
    )
    summary = summarize(rows, stats.min_samples_p95)
    (run_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    return summary


def run_command(args: argparse.Namespace) -> int:
    plan = _build_plan(args)
    print(_plan_text(plan))
    if args.dry_run:
        print("dry run: no requests sent")
        return 0

    api_key = load_api_key(args.env_file)
    redact = Redactor((api_key,))
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(_metadata(run_id, plan, args.models_csv), indent=2), encoding="utf-8"
    )
    print(f"run {run_id}: writing to {run_dir} (seed {plan.seed}, concurrency {plan.concurrency})")

    with (
        (run_dir / "raw.csv").open("w", newline="", encoding="utf-8") as handle,
        local_tracing(run_dir / "traces.jsonl", redact),
    ):
        writer = RawWriter(handle)
        try:
            outcome = asyncio.run(_benchmark(run_id, plan, api_key, writer, redact))
        finally:
            # Runs on a crash or Ctrl-C too, so the rows so far still get reports.
            summary = _write_reports(run_dir, writer.rows, plan.settings)

    print(f"\n{summary}\n")
    print(
        f"scenarios completed: {outcome.completed}, aborted: {outcome.aborted}, not sent: {outcome.not_sent}"
        + (f", rejected pairs: {', '.join(outcome.rejected_pairs)}" if outcome.rejected_pairs else "")
    )
    print(_files_line(run_dir))
    if outcome.fatal_error:
        print(f"run stopped early: {outcome.fatal_error}", file=sys.stderr)
        return 1
    return 0


def combine_command(args: argparse.Namespace) -> int:
    settings = load_settings(args.settings)
    rows = combine_runs(args.run_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "raw.csv", RAW_COLUMNS, rows)
    summary = _write_reports(args.output_dir, rows, settings)
    sources = {"runs": sorted({row["run_id"] for row in rows}), "run_dirs": [str(path) for path in args.run_dirs]}
    (args.output_dir / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(f"{summary}\n\n{_files_line(args.output_dir)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = run_command if args.command == "run" else combine_command
    try:
        return command(args)
    except (
        CombineError,
        ModelsCsvError,
        SettingsError,
        CredentialsError,
        FileNotFoundError,
        FileExistsError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "interrupted: raw.csv keeps every finished attempt; reports cover the rows written so far", file=sys.stderr
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())
