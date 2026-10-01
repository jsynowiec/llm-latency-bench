"""End-to-end runs of the CLI against a local fake of OpenRouter. No network access."""

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import pytest

from llm_latency_bench import cli, openrouter
from tests.conftest import FAKE_API_KEY, REPO_ROOT
from tests.fake_openrouter import FakeOpenRouter, Script, error_reply, text_reply

SINGLE_TURN = ("turn_counts = [1, 5, 10]", "turn_counts = [1]")
# The pair with a single reasoning level in models.csv, so filtered runs stay small.
ONE_PAIR = ("--model", "openai/gpt-6-luna")
CONTEXT_LABELS = ["0.25k", "2.5k", "10k", "32k", "64k"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def install_fake(monkeypatch, script: Script) -> FakeOpenRouter:
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_API_KEY)
    fake = FakeOpenRouter(script)
    real_build = openrouter.build_http_client
    monkeypatch.setattr(cli, "build_http_client", lambda settings: real_build(settings, transport=fake.transport()))
    return fake


@pytest.fixture
def fake_openrouter(monkeypatch) -> FakeOpenRouter:
    def script(body, number):
        # Every tenth request fails once with an error that echoes the API key back.
        if number % 10 == 0:
            return error_reply(503, f"upstream said {FAKE_API_KEY}")
        return text_reply(f"Answer {number}.")

    return install_fake(monkeypatch, script)


def settings_file(tmp_path: Path, name: str, *edits: tuple[str, str], base: str = "settings.toml") -> Path:
    """A copy of a committed settings file with text edits, and without real backoff waits."""
    text = (REPO_ROOT / base).read_text(encoding="utf-8")
    for old, new in [("backoff_base_s = 2.0", "backoff_base_s = 0.0"), *edits]:
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / f"{name}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def run_cli(tmp_path: Path, *extra: str, settings: Path | None = None, output_dir: Path | None = None) -> int:
    return cli.main(
        [
            "run",
            "--models-csv",
            str(REPO_ROOT / "models.csv"),
            "--settings",
            str(settings or settings_file(tmp_path, "settings")),
            "--env-file",
            str(tmp_path / "missing.env"),
            "--output-dir",
            str(output_dir or tmp_path / "outputs"),
            *extra,
        ]
    )


def only_run_dir(output_dir: Path) -> Path:
    [run_dir] = output_dir.iterdir()
    return run_dir


def finished_run(tmp_path: Path, name: str, *edits: tuple[str, str], base: str = "settings.toml") -> Path:
    """The folder of a successful one-pair run with the given settings edits."""
    settings = settings_file(tmp_path, name, *edits, base=base)
    assert run_cli(tmp_path, *ONE_PAIR, settings=settings, output_dir=tmp_path / name) == 0
    return only_run_dir(tmp_path / name)


def combine(*run_dirs: Path, output_dir: Path) -> int:
    return cli.main(["combine", *map(str, run_dirs), "--output-dir", str(output_dir)])


@pytest.mark.parametrize(
    ("edits", "extra", "scenarios", "requests"),
    [
        ((), ("--repetitions", "2"), 432, "2312 (2304 scenario turns + 8 warm-up)"),
        ((("warmup = true", "warmup = false"),), ONE_PAIR, 27, "144 (144 scenario turns + 0 warm-up)"),
    ],
    ids=["with-warm-up", "without-warm-up"],
)
def test_dry_run_prints_the_plan_and_sends_nothing(
    tmp_path, capsys, fake_openrouter, edits, extra, scenarios, requests
):
    settings = settings_file(tmp_path, "settings", *edits)
    assert run_cli(tmp_path, "--dry-run", *extra, settings=settings) == 0

    text = capsys.readouterr().out
    assert f"planned scenarios: {scenarios} " in text
    assert f"expected requests: {requests}, excluding retries" in text
    assert fake_openrouter.requests == []
    assert not (tmp_path / "outputs").exists()


def test_invalid_models_csv_exits_with_an_error(tmp_path, capsys):
    bad = tmp_path / "models.csv"
    bad.write_text("model_name,model_slug\nA,a\n", encoding="utf-8")

    assert cli.main(["run", "--models-csv", str(bad), "--settings", str(REPO_ROOT / "settings.toml")]) == 2
    assert "missing required column" in capsys.readouterr().err


def test_smoke_run_writes_both_csvs_and_never_leaks_credentials(tmp_path, capsys, fake_openrouter):
    exit_code = run_cli(tmp_path, "--model", "openai/gpt-6-sol", "--reasoning", "high", "--seed", "3")

    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    run_dir = only_run_dir(tmp_path / "outputs")
    raw = read_csv(run_dir / "raw.csv")
    aggregated = read_csv(run_dir / "aggregated.csv")

    # 1 warm-up + 144 scenario turns, plus one retry for every tenth request.
    assert len(fake_openrouter.requests) == len(raw)
    assert sum(row["status"] == "ok" for row in raw) == 145
    assert {row["retry_index"] for row in raw if row["status"] == "error"} == {"0"}
    assert len(aggregated) == 27
    assert all(row["successful_samples"] == "1" and row["failure_count"] == "0" for row in aggregated)
    # One sample per cell, so tail percentiles are flagged. The fake reports zero cached tokens, recorded as 0.
    assert all(row["p95_low_confidence"] == "true" for row in aggregated)
    assert all(row["cache_read_tokens_p50"] == "0.000000" for row in aggregated)
    assert {row["turn_count"] for row in aggregated} == {"1", "5", "10"}

    trace_text = (run_dir / "traces.jsonl").read_text(encoding="utf-8")
    spans = [json.loads(line) for line in trace_text.splitlines()]
    names = Counter(span["name"] for span in spans)
    assert names["attempt {scenario_id} turn {turn_index} retry {retry_index}"] == len(raw)
    assert names["chat openai/gpt-6-sol"] == len(raw)
    scenario_ids = {span["attributes"]["scenario_id"] for span in spans if span["name"] == "scenario {scenario_id}"}
    assert scenario_ids == {row["scenario_id"] for row in raw if row["phase"] == "measure"}
    # Spans leave out message text: no reply, user prompt, or reference context.
    assert "Answer 1." not in trace_text
    assert "asset_register" not in trace_text

    outputs = [captured.out, captured.err, trace_text, *(path.read_text() for path in run_dir.iterdir())]
    for text in outputs:
        assert FAKE_API_KEY not in text
    assert "[REDACTED]" in (run_dir / "raw.csv").read_text()
    assert (run_dir / "summary.txt").read_text(encoding="utf-8") in captured.out
    assert "scenarios completed: 27, aborted: 0, not sent: 0\n" in captured.out
    assert f"files: {run_dir / 'raw.csv'}, {run_dir / 'aggregated.csv'}, {run_dir / 'summary.txt'}" in captured.out


@pytest.mark.parametrize("option", ["--repetitions", "--concurrency"])
def test_zero_repetitions_or_concurrency_is_rejected(tmp_path, capsys, option):
    assert run_cli(tmp_path, "--dry-run", option, "0") == 2
    assert "at least 1" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("status", "failing_request", "category", "completed"),
    [(401, 1, "auth", 0), (402, 3, "insufficient_credits", 1)],
    ids=["auth-at-warm-up", "credits-mid-run"],
)
def test_auth_or_credit_failure_stops_the_run_with_exit_code_1(
    tmp_path, capsys, monkeypatch, status, failing_request, category, completed
):
    def script(body, number):
        return error_reply(status, "no") if number == failing_request else text_reply("Fine.")

    fake = install_fake(monkeypatch, script)
    settings = settings_file(tmp_path, "settings", SINGLE_TURN)

    # One scenario at a time, so the request that fails is known in advance.
    assert run_cli(tmp_path, *ONE_PAIR, "--concurrency", "1", settings=settings) == 1

    assert f"run stopped early: {category} (HTTP {status})" in capsys.readouterr().err
    assert len(fake.requests) == failing_request
    run_dir = only_run_dir(tmp_path / "outputs")
    measure = [row for row in read_csv(run_dir / "raw.csv") if row["phase"] == "measure"]
    not_sent = [row for row in measure if row["status"] == "not_sent"]
    # Request 1 is the warm-up and request n the scenario in position n - 1; every later scenario is never sent.
    assert sorted(int(row["scenario_order"]) for row in not_sent) == list(range(failing_request, 10))
    assert {row["error_category"] for row in not_sent} == {"run_aborted"}
    aggregated = read_csv(run_dir / "aggregated.csv")
    assert len(aggregated) == 9
    assert sum(int(row["successful_samples"]) for row in aggregated) == completed
    assert "== Run totals ==" in (run_dir / "summary.txt").read_text(encoding="utf-8")


def stop_after_first_scenario(monkeypatch, error: BaseException) -> None:
    real_run_matrix = cli.run_matrix

    async def run_one_then_fail(runner, pairs, scenarios, **kwargs):
        await real_run_matrix(runner, pairs, scenarios[:1], **kwargs)
        raise error

    monkeypatch.setattr(cli, "run_matrix", run_one_then_fail)


def assert_reports_cover_the_first_scenario(run_dir: Path) -> None:
    [final] = [
        row for row in read_csv(run_dir / "raw.csv") if row["phase"] == "measure" and row["is_final_turn"] == "true"
    ]
    [cell] = read_csv(run_dir / "aggregated.csv")
    assert cell["scenario_runs"] == cell["successful_samples"] == "1"
    for column in ("prompt_size", "context_size", "turn_count"):
        assert cell[column] == final[column]
    assert cell["ttft_s_p50"] == final["ttft_s"]
    assert "scenario runs: 1, valid final-turn samples: 1," in (run_dir / "summary.txt").read_text()


def test_crash_mid_run_still_writes_reports_for_the_rows_so_far(tmp_path, fake_openrouter, monkeypatch):
    stop_after_first_scenario(monkeypatch, RuntimeError("simulated crash"))

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_cli(tmp_path, *ONE_PAIR)

    assert_reports_cover_the_first_scenario(only_run_dir(tmp_path / "outputs"))


def test_ctrl_c_exits_130_and_still_writes_reports(tmp_path, capsys, fake_openrouter, monkeypatch):
    stop_after_first_scenario(monkeypatch, KeyboardInterrupt())

    assert run_cli(tmp_path, *ONE_PAIR) == 130

    assert "interrupted" in capsys.readouterr().err
    assert_reports_cover_the_first_scenario(only_run_dir(tmp_path / "outputs"))


def strip_target_token_columns(run_dir: Path) -> None:
    """Rewrite raw.csv as a run written before raw.csv had target-token columns."""
    rows = read_csv(run_dir / "raw.csv")
    columns = [name for name in rows[0] if not name.endswith("_target_tokens")]
    with (run_dir / "raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_combine_merges_runs_relabels_context_by_tokens_and_fills_old_columns(tmp_path, capsys, fake_openrouter):
    old = finished_run(tmp_path, "old", SINGLE_TURN)
    # The label "short" means 250 tokens in the old run and 32,000 here, so only relabelling keeps them apart.
    extra = finished_run(
        tmp_path, "extra", SINGLE_TURN, ("short = 250\nmedium = 2500\nlong = 10000", "short = 32000\nlong = 64000")
    )
    strip_target_token_columns(old)
    old_raw = (old / "raw.csv").read_bytes()
    capsys.readouterr()

    out = tmp_path / "combined"
    assert combine(old, extra, output_dir=out) == 0

    aggregated = read_csv(out / "aggregated.csv")
    assert len(aggregated) == 3 * 5
    for prompt_size in ("short", "medium", "long"):
        contexts = [row["context_size"] for row in aggregated if row["prompt_size"] == prompt_size]
        assert contexts == CONTEXT_LABELS
    sizes = {(row["context_size"], row["context_target_tokens"]) for row in aggregated}
    assert sizes == {("0.25k", "250"), ("2.5k", "2500"), ("10k", "10000"), ("32k", "32000"), ("64k", "64000")}
    assert all(row["scenario_runs"] == row["successful_samples"] == "1" for row in aggregated)

    filled = [row for row in read_csv(out / "raw.csv") if row["run_id"] == old.name and row["phase"] == "measure"]
    assert len({row["scenario_id"] for row in filled}) == 9
    prompt_targets = {"short": "40", "medium": "400", "long": "2000"}
    assert all(row["prompt_target_tokens"] == prompt_targets[row["prompt_size"]] for row in filled)
    assert {row["context_target_tokens"] for row in filled} == {"250", "2500", "10000"}

    summary = (out / "summary.txt").read_text(encoding="utf-8")
    header = next(line for line in summary.splitlines() if "TTFT/reasoning" in line)
    assert header.split()[2::3] == CONTEXT_LABELS
    assert json.loads((out / "sources.json").read_text()) == {
        "runs": sorted([old.name, extra.name]),
        "run_dirs": [str(old), str(extra)],
    }
    # The run folders are inputs only.
    assert (old / "raw.csv").read_bytes() == old_raw


def test_combining_repeats_of_one_setup_pools_the_final_turns_of_every_run(tmp_path, capsys, fake_openrouter):
    two_turn_counts = ("turn_counts = [1, 5, 10]", "turn_counts = [1, 3]")
    first, second = finished_run(tmp_path, "first", two_turn_counts), finished_run(tmp_path, "second", two_turn_counts)

    assert combine(first, second, output_dir=tmp_path / "combined") == 0

    final_ttft: dict[str, list[float]] = {}
    for run_dir in (first, second):
        for row in read_csv(run_dir / "raw.csv"):
            if row["is_final_turn"] == "true" and row["status"] == "ok" and row["phase"] == "measure":
                cell = "/".join((row["prompt_size"], row["context_target_tokens"], row["turn_count"]))
                final_ttft.setdefault(cell, []).append(float(row["ttft_s"]))
    aggregated = read_csv(tmp_path / "combined" / "aggregated.csv")
    assert len(aggregated) == len(final_ttft) == 3 * 3 * 2
    for row in aggregated:
        # Both runs share every scenario ID, and only their final turns are samples.
        assert (row["scenario_runs"], row["successful_samples"], row["failure_count"]) == ("2", "2", "0")
        cell = "/".join((row["prompt_size"], row["context_target_tokens"], row["turn_count"]))
        assert float(row["ttft_s_p50"]) == pytest.approx(statistics.median(final_ttft[cell]), abs=1e-6)


def test_combine_refuses_a_run_listed_twice_even_when_not_adjacent(tmp_path, capsys, fake_openrouter):
    first, second = finished_run(tmp_path, "first", SINGLE_TURN), finished_run(tmp_path, "second", SINGLE_TURN)
    capsys.readouterr()

    assert combine(first, second, first, output_dir=tmp_path / "combined") == 2
    assert f"{first}: run {only_run_dir(tmp_path / 'first').name} is listed twice" in capsys.readouterr().err
    assert not (tmp_path / "combined").exists()


def test_combine_refuses_a_prompt_label_with_different_target_sizes(tmp_path, capsys, fake_openrouter):
    base = finished_run(tmp_path, "base", SINGLE_TURN)
    bigger = finished_run(tmp_path, "bigger", SINGLE_TURN, ("short = 40", "short = 80"))
    capsys.readouterr()

    assert combine(base, bigger, output_dir=tmp_path / "combined") == 2
    assert "prompt size 'short' is 80 tokens here but 40 elsewhere" in capsys.readouterr().err
    assert not (tmp_path / "combined").exists()


def test_combine_never_writes_into_an_existing_folder(tmp_path, capsys, fake_openrouter):
    run_dir = finished_run(tmp_path, "run", SINGLE_TURN)
    out = tmp_path / "combined"
    out.mkdir()
    (out / "notes.txt").write_text("keep me", encoding="utf-8")
    capsys.readouterr()

    assert combine(run_dir, output_dir=out) == 2
    assert "File exists" in capsys.readouterr().err
    assert [path.name for path in out.iterdir()] == ["notes.txt"]
    assert (out / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_combine_reports_a_size_label_missing_from_old_metadata(tmp_path, capsys, fake_openrouter):
    run_dir = finished_run(tmp_path, "run", SINGLE_TURN)
    strip_target_token_columns(run_dir)
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    del metadata["settings"]["sizes"]["context_tokens"]["long"]
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata))
    capsys.readouterr()

    assert combine(run_dir, output_dir=tmp_path / "combined") == 2
    assert "size 'long' in raw.csv is not in run_metadata.json" in capsys.readouterr().err


def test_settings_of_the_wrong_type_stop_the_run_before_any_request(tmp_path, capsys, fake_openrouter):
    settings = settings_file(tmp_path, "bad", ("concurrency = 2", "concurrency = 2.5"))

    assert run_cli(tmp_path, settings=settings) == 2
    assert "run.concurrency must be a int" in capsys.readouterr().err
    assert fake_openrouter.requests == []
    assert not (tmp_path / "outputs").exists()


def test_combine_refuses_runs_with_different_output_settings(tmp_path, capsys, fake_openrouter):
    short = finished_run(tmp_path, "short", SINGLE_TURN)
    long = finished_run(tmp_path, "long", SINGLE_TURN, ("answer_words = 80", "answer_words = 200"))
    capsys.readouterr()

    assert combine(short, long, output_dir=tmp_path / "combined") == 2
    assert "output settings" in capsys.readouterr().err
    assert not (tmp_path / "combined").exists()
