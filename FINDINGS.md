# Findings: Gemini 2.5 Flash Load Test Results

## Workload

`load_test.py` cycles through four short, realistic prompts (customer
support triage, technical doc summarization, code review advice, and a
scheduling email draft) against `gemini-2.5-flash` via Vertex AI's
`generateContent` REST endpoint, at `temperature=0.7`. It sweeps concurrency
levels `[1, 5, 10, 20, 30, 50, 75, 100]`, running `max(20, concurrency)`
requests per level through an `asyncio.Queue` + worker-pool pattern, with a
3-second cooldown between levels.

## Results

**Zero failures at every concurrency level, 1 through 100** — 335 requests
total, all successful.

**Latency is bimodal and flat across concurrency**: p50 sits at ~3-5s at
every level, but p90/p95/p99 sit at ~30-45s at every level too — including
concurrency=1, fully sequential, zero contention.

| concurrency | reqs | rps | p50 | p90 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 0.09 | 4.5s | 30.3s | 30.7s | 31.1s | 10.7s |
| 5 | 20 | 0.38 | 3.2s | 31.9s | 37.3s | 44.7s | 10.5s |
| 10 | 20 | 0.43 | 3.4s | 32.7s | 42.8s | 43.8s | 10.9s |
| 20 | 20 | 0.53 | 4.6s | 29.9s | 32.6s | 36.6s | 10.5s |
| 30 | 30 | 0.75 | 4.1s | 31.4s | 32.9s | 37.8s | 9.8s |
| 50 | 50 | 1.42 | 4.1s | 31.9s | 34.5s | 35.1s | 9.9s |
| 75 | 75 | 1.87 | 4.7s | 33.0s | 36.1s | 38.8s | 11.2s |
| 100 | 100 | 2.57 | 4.1s | 33.9s | 35.8s | 37.3s | 10.7s |

Full per-scenario data, including throughput and token-rate figures, is in
`load_test_results.json`.

## Retry telemetry

Every request in this run now reports how many attempts it took (via a new
`attempt_number` field on the response). Aggregated per scenario in
`load_test_results.json` as `attempt_count_breakdown`, plus separate latency
percentiles for single-attempt vs. multi-attempt requests.

Across all 335 requests, at every concurrency level tested:

- `attempt_count_breakdown` is `{"1": N}` — every request succeeded on the
  first attempt.
- `multi_attempt_requests` is `0` at every concurrency level.
- `single_attempt_latency_p50/p90/p95/p99` are identical to the overall
  latency percentiles in the table above (since 100% of requests are
  single-attempt).

No retries fired during this run. The bimodal p50/p90+ latency tail
observed above is not associated with retry activity — every request in
both the fast (~3-5s) and slow (~30-45s) portions of the distribution
completed on its first attempt.
