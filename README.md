# llm-latency-bench

llm-latency-bench measures the streaming latency of large language models on [OpenRouter](https://openrouter.ai). It records the time to first token (TTFT), the output speed in tokens per second (TPS), and the total time of each request. It writes one CSV row for each request attempt and one CSV row for each test cell.

## Setup

You need [uv](https://docs.astral.sh/uv/) and an OpenRouter API key.

1. Install the dependencies: `uv sync`.
2. Copy `.env.example` to `.env` in the repository root.
3. Put your key in `OPENROUTER_API_KEY` in `.env`. Do not commit `.env`.
4. Edit `models.csv` with model/reasoning pairs you want to test.

The tool stops before the first request if the key is missing or is still the placeholder from `.env.example`.

Each row of `models.csv` must have `model_name`, `reasoning_level_name`, `model_slug`, and `reasoning_level`. The `model_slug` is any OpenRouter model ID, in the form `provider/model`, for example `openai/gpt-6-sol`. If a cell is blank or a `model_slug` and `reasoning_level` pair repeats, the tool stops and names the line.

## Run the benchmark

Print the plan first. This command sends no requests:

```sh
uv run llm-latency-bench run --dry-run
```

Run the full matrix from `models.csv`:

```sh
uv run llm-latency-bench run --models-csv models.csv --output-dir outputs --repetitions 1 --seed 20260930 --concurrency 2
```

The context sweep measures where the input size starts to show in TTFT. It uses single-turn scenarios only, so no request can read a cached prefix. It runs context sizes of about 2,000, 16,000, and 128,000 tokens, 5 times each. A second file adds points at 32,000 and 64,000 tokens:

```sh
uv run llm-latency-bench run --settings settings.context-sweep.toml --output-dir outputs/context-sweep
uv run llm-latency-bench run --settings settings.context-sweep-32k-64k.toml --output-dir outputs/context-sweep
```

To see several runs in one report, combine them. The command reads `raw.csv` and `run_metadata.json` in each run folder. It writes `raw.csv`, `aggregated.csv`, `summary.txt`, and `sources.json` to a new folder and does not change the run folders:

```sh
uv run llm-latency-bench combine outputs/context-sweep/<run id> outputs/context-sweep/<run id> --output-dir outputs/context-sweep/combined
```

In combined files, each context size gets a label from its target size, for example `16k`. A prompt size keeps its label, and that label must have the same target size in every run. All runs must have the same `[output]` settings.

| Option | Default | Meaning |
|---|---|---|
| `--models-csv PATH` | `models.csv` | The matrix of models and reasoning levels. |
| `--settings PATH` | `settings.toml` | Timeouts, retries, sizes, and other values that are not secret. |
| `--env-file PATH` | `.env` | The file with `OPENROUTER_API_KEY`. |
| `--output-dir DIR` | `outputs` | The tool writes to `DIR/<run id>/`. It never overwrites an earlier run. |
| `--repetitions N` | `run.repetitions` | The number of times each scenario runs. |
| `--seed N` | `run.seed` | The seed for the random order of the scenarios. |
| `--concurrency N` | `run.concurrency` (2) | The number of scenarios that run at the same time. |
| `--model SLUG` | all | Run only this `model_slug`. You can repeat the option. |
| `--reasoning LEVEL` | all | Run only this `reasoning_level`. You can repeat the option. |
| `--dry-run` | off | Print the plan and stop. |

## Experimental design

For each pair of model and reasoning level, the tool runs every combination of three factors. `settings.toml` uses these levels:

| Factor | Levels | Target size |
|---|---|---|
| Prompt size: the latest user message | short, medium, long | about 40, 400, and 2,000 tokens |
| Context size: a fixed reference text in the system message | short, medium, long | about 250, 2,500, and 10,000 tokens |
| Turn count: the number of turns in one conversation | 1, 5, 10 | not applicable |

This gives 27 scenarios and 144 requests for each pair and repetition.

The reference context is a made-up asset register for a water utility. The prompts are questions about the register. A medium prompt adds a short brief, and a long prompt also adds pasted field notes. The text comes from a fixed seed, so every run sends the same text. The tool sizes text at 4 characters per token. The `input_tokens` column holds the real size that the provider reports.

In a multi-turn scenario, the tool sends the real conversation: the system message, each earlier question, and each earlier answer. The final turn is the main comparison. The final turn of every scenario sends the same prompt, so the final turns of a 1-turn, 5-turn, and 10-turn scenario differ only in the history before them. There are 10 questions, so the turn count cannot be more than 10.

The tool shuffles all scenarios with the seed, so one model does not always run at the same time of day. Turns inside a scenario always run in order.

### Warm-up and caching

Before the scenarios, the tool sends one short warm-up request for each pair. The warm-up opens the connection and makes sure that OpenRouter accepts the pair. The aggregates do not use warm-up rows. The summary shows the warm-up TTFT next to the measured turn-1 TTFT, so you can see a cold start.

The system message starts with a random session ID for each scenario, so one scenario cannot reuse the prompt cache of another. Inside one conversation, later turns can read earlier turns from the cache, as in real use. The `cache_read_tokens` column shows how much of each input came from the cache. The summary also shows the turn-1 TTFT of the 5-turn and 10-turn scenarios. That first request is the same in both, so a difference between them shows the noise level of the run.

## Metrics

All durations come from `time.perf_counter`, a clock that only moves forward.

| Metric | Definition |
|---|---|
| `ttft_s` | From just before the request starts to the first stream event with visible text that is not only whitespace. Empty chunks, usage chunks, and reasoning do not count. |
| `post_ttft_s` | From the first visible text to the end of the stream, including the final usage chunk. |
| `tps` | Visible output tokens divided by `post_ttft_s`. Visible output tokens are `output_tokens` minus `reasoning_tokens`, as the provider reports them. If either count is missing, `tps` is empty. The tool never estimates tokens from characters. |
| `request_latency_s` | From the request start to the end of the stream, or to the failure. |
| `scenario_total_s` | Wall-clock time for all turns of the scenario, including retry waits (`scenario_backoff_s`). It is on the last row of the scenario. |

## Failures and retries

The tool retries a turn itself, up to `retries.max_retries` (2 by default) times. The wait doubles after each retry, up to `retries.backoff_max_s`, and a random factor from 0.5 to 1 shortens it. Each attempt gets its own row with `retry_index`. The tool retries only these errors: `timeout` (including HTTP 408), `rate_limited` (HTTP 429), `server_error` (HTTP 5xx), `connection`, and `stream_error` (the stream broke, ended without a finish reason, or ended with `finish_reason` = `error`).

The tool does not retry `bad_request` (HTTP 400 or 422), `auth` (401), `insufficient_credits` (402), `forbidden` (403, a guardrail or moderation block), `not_found` (404), `http_error` (any other status), `truncated` (the stream stopped at `max_tokens`), `content_filter`, `no_visible_output`, `unexpected_finish`, and `other`. If a turn still fails after its retries, the scenario stops, because the next turn needs the missing answer.

If the warm-up of a pair gets HTTP 400, 403, 404, or 422, the tool does not run that pair. It writes one `not_sent` row with `error_category` = `pair_rejected` for each planned scenario of the pair. If a request gets HTTP 401 or 402, the tool stops the run, writes `not_sent` rows with `run_aborted` for the scenarios that did not start, writes the reports, and exits with code 1.

## Output files

Each run writes to `outputs/<run id>/`. An empty cell means that no value exists. It never means zero.

| File | Content |
|---|---|
| `raw.csv` | One row for each request attempt. The tool writes each row when the attempt ends, so an interrupted run keeps its data. |
| `aggregated.csv` | One row for each model, reasoning level, prompt size, context size, and turn count. |
| `summary.txt` | The summary that the tool prints at the end. |
| `traces.jsonl` | Logfire spans, one JSON object on each line. |
| `run_metadata.json` | The seed, the options, the settings, the selected pairs, and the package versions. |

### raw.csv columns

| Column | Meaning |
|---|---|
| `run_id`, `seed`, `phase`, `request_seq`, `started_at_utc` | The run, its seed, `warmup` or `measure`, the send order across the run, and the UTC start time of the attempt. |
| `scenario_id`, `scenario_order`, `repetition` | The scenario, its position in the shuffled order, and its repetition number. |
| `model_name`, `model_slug`, `reasoning_level_name`, `reasoning_level` | The values from `models.csv`. |
| `prompt_size`, `prompt_target_tokens`, `context_size`, `context_target_tokens`, `turn_count`, `turn_index`, `is_final_turn` | The scenario cell, its target sizes in tokens, and the turn (1-based). |
| `retry_index`, `status` | 0 for the first attempt of a turn. `status` is `ok`, `error`, or `not_sent`. |
| `error_category`, `http_status`, `error_message` | The failure. The tool removes the API key from the message and limits it to 500 characters. |
| `finish_reason` | `stop`, `length`, `content_filter`, `tool_call`, or `error`, as Pydantic AI reports it. Only `stop` gives a valid sample. |
| `response_model`, `upstream_provider`, `generation_id` | The model name in the response, the provider that OpenRouter sent the request to, and the OpenRouter generation ID. |
| `input_tokens`, `cache_read_tokens`, `cache_write_tokens`, `output_tokens`, `reasoning_tokens`, `visible_output_tokens` | Token counts from the provider. They are empty if the provider sent no usage data. |
| `visible_chars` | The number of visible characters that the tool received. |
| `ttft_s`, `post_ttft_s`, `request_latency_s`, `tps` | The timings. See [Metrics](#metrics). |
| `scenario_status`, `scenario_total_s`, `scenario_backoff_s`, `scenario_started_at_utc`, `scenario_ended_at_utc` | Only on the last row of a scenario. `scenario_status` is `completed`, `aborted`, or `not_sent`. |

You can look up a request in the OpenRouter activity log or at `GET /api/v1/generation?id=<generation_id>`.

### aggregated.csv columns

The aggregates use only the final turn of each scenario. A sample is a final-turn attempt with `status` = `ok`.

| Column | Meaning |
|---|---|
| `model_name` to `turn_count` | The cell, including the target sizes. Rows are sorted by target size. |
| `scenario_runs`, `successful_samples` | The number of scenario runs and samples in the cell. |
| `failure_count`, `failure_rate` | Scenario runs without a sample, and their share of `scenario_runs`. |
| `failed_attempts`, `retried_scenarios` | All `error` rows in the cell on any turn, including errors that a retry fixed, and the scenario runs with at least one retry. |
| `tps_samples` | Samples with a `tps` value. |
| `<metric>_p10` to `<metric>_p99` | Percentiles 10, 25, 50, 75, 95, and 99 of `ttft_s`, `request_latency_s`, `tps`, and `scenario_total_s`. |
| `<tokens>_p50` | The median of each token count. |
| `p95_low_confidence`, `p99_low_confidence` | `true` if the cell has fewer than 20 (p95) or 100 (p99) samples. |

The percentiles use linear interpolation between the closest ranks (Hyndman and Fan type 7, the numpy default).

## Limits

- Each repetition gives one sample for each cell. A p95 needs about 20 samples in a cell, and a p99 needs about 100. For comparisons across one factor, the summary pools the other two factors, which gives 9 times more samples for each level.
- The summary shows measurements only. A TTFT difference between sizes does not prove a cause. Queues, caching, and provider load also change TTFT.
- The tool sends `reasoning_level` unchanged as `reasoning.effort`, with `reasoning.exclude` = `true`. If a model does not accept a level, the request fails and the tool records the failure. Because reasoning is excluded, TTFT includes the reasoning time.
- `max_tokens` (32,000) includes reasoning tokens.
- OpenRouter picks an upstream provider for each request and can fall back to another provider if the first one fails. The `upstream_provider` column shows the provider that answered. A fallback inside one request is not visible in the stream, so its time is part of the measured latency.
- OpenRouter can send keep-alive comments while a model reasons. Then `client.read_timeout_s` does not trigger, and `client.request_timeout_s` is the effective limit.
- `tps` counts the tokens of the first visible chunk, which arrive at TTFT, so it reads slightly high for short answers.
- OpenRouter or the provider can buffer the answer and send it in one burst. If `post_ttft_s` is only a few milliseconds, `tps` measures delivery speed, not generation speed.

## Traces

Logfire writes spans to `traces.jsonl` in the run folder. It sends no data to the Logfire service and needs no Logfire account. The spans do not contain message text, and the tool removes the API key from every span before it writes the span. If you set `OTEL_EXPORTER_OTLP_ENDPOINT`, Logfire also sends the spans to that OpenTelemetry endpoint. Spans sent this way get only the Logfire scrubbing, which does not cover exception messages.

## Development

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

The tests use a local fake of the OpenRouter endpoint (an `httpx2.MockTransport` that streams server-sent events with delays). They send no network requests.

## License

MIT. See [LICENSE](LICENSE).
