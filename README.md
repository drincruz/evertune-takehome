# Evertune Take-home Exercise: Adding Gemini 2.5 Flash

This repo contains a small sample of our LLM vendor integration. We'd like you to add support for Gemini 2.5 Flash on Google Vertex and report back on your findings.

# Setup

You'll need the `gcloud` CLI installed and configured against our project, which we will provide for you. The Gemini provider authenticates via Application Default Credentials, so you'll also need to run:

```bash
gcloud auth application-default login
```

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync              # install dependencies into .venv
uv run pytest        # run the test suite
uv run python load_test.py   # run the load test
```

## Environment variables

### Gemini / Vertex AI

| Variable | Required | Default | Notes |
|---|---|---|---|
| `VERTEX_PROJECT` | Yes | *(none)* | GCP project ID to call Vertex AI in. `Gemini()` raises `ValueError` at construction time if this isn't set and no `project=` is passed explicitly — there's no implicit default project. |
| `VERTEX_LOCATION` | No | `us-central1` | Vertex AI region. |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Model ID passed to the `generateContent` endpoint. |
| `GEMINI_MAX_RETRIES` | No | `5` | Max attempts (including the first) for retryable errors (429/500/502/503/504, plus transport errors). |
| `GEMINI_BACKOFF_MAX_SECONDS` | No | `20` | Ceiling for the jittered exponential backoff between retries. |
| `GEMINI_PARALLELISM` | No | `100` | Suggested concurrency, returned by `Gemini.parallelism()`. Advisory only — the library doesn't enforce it; callers (like `load_test.py`) are responsible for actually limiting concurrency. |

### Logging

| Variable | Required | Default | Notes |
|---|---|---|---|
| `LOG_LEVEL` | No | `INFO` | Used by `configure_logging()` (`llm/log_config.py`), which `load_test.py` calls on startup. Writes JSON logs to `tmp/load_test.log`. |

### Together AI provider

`load_test.py` only exercises `Gemini`, so these aren't needed to run it — but they're required if you instantiate `llm.Together` directly.

| Variable | Required | Default | Notes |
|---|---|---|---|
| `TOGETHER_API_KEY` | Yes (for `Together`) | *(none)* | API key for Together AI. |
| `TOGETHER_MODEL` | Yes (for `Together`) | *(none)* | Model ID to use. |

# What to build

Implement Gemini 2.5 Flash as a provider in this system, and demonstrably prove it will hold up at production scale. We care about both halves of that sentence: a working integration *and* the evidence that it will not fall over when we point real traffic at it.

How you structure the code is up to you — the existing providers are a reference, not a template. If something about Gemini doesn't fit those patterns, deviate and tell us why.

For the "prove it works at scale" half: design and run whatever load tests, harnesses, or experiments you'd want to see before signing off on this for production. Show us the numbers, the failure modes you uncovered, and the headroom (or lack thereof) you found.

# Deliverables

We're less interested in a "completed checklist" and more interested in what you learned. In your write-up, we'd like to see:

- How the integration behaves under realistic load. Pick a workload, run it, and tell us what you observed.
- Anything you discovered about this model — quirks, failure modes, parameters that mattered, things that surprised you compared to other LLMs you've used.
- Decisions you made and the tradeoffs behind them. If you tried something that didn't work, that's worth including too.
- What you'd want to do next if this were going to production, and what you'd want to know before getting there.