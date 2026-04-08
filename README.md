---
title: gREV
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: RL sandbox for autonomous coding agents.
tags:
  - openenv
  - reinforcement-learning
  - agents
  - devops
  - python
  - hackathon
---

# gREV

> An **OpenEnv-compliant** environment where AI agents are dropped into broken Python repositories and must diagnose and fix them — using real shell commands and file edits — until the full `pytest` suite passes.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-7C3AED)](https://github.com/openenv)
[![HF Space](https://img.shields.io/badge/🤗_Space-live-blue)](https://langersword-grev-openenv.hf.space)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688)](https://fastapi.tiangolo.com)

**Live endpoint:** `https://langersword-grev-openenv.hf.space`
**Source code:** `github.com/LangerSword/gREV`
**Hackathon:** Scaler × Meta / Hugging Face OpenEnv Challenge — April 2026

---

## Table of contents

- [Overview](#overview)
- [Why this domain](#why-this-domain)
- [Architecture](#architecture)
- [Observation space](#observation-space)
- [Action space](#action-space)
- [Tasks](#tasks)
- [Reward function](#reward-function)
- [Setup](#setup)
- [Running the baseline agent](#running-the-baseline-agent)
- [API reference](#api-reference)
- [Baseline scores](#baseline-scores)
- [Validation](#validation)

---

## Overview

Every software engineer knows the experience: CI goes red, `pytest` is failing, and you need to find and fix the bug before the next deploy. gREV turns this into a structured training environment for AI agents.

The agent is given a **broken Python repository** with one or more bugs deliberately introduced. It has access to a sandboxed shell — it can run `pytest`, `cat` files, `grep` for patterns — and it can overwrite files with corrected content. The episode ends when all tests pass (reward 1.0) or the step budget runs out (fractional credit based on how many tests it fixed).

Every aspect of grading is **fully deterministic** — scores are computed by parsing `pytest`'s structured output, never by an LLM judge or human opinion. The same broken repo, same seed, same actions always produce the same score.

---

## Why this domain

Software debugging is one of the most time-intensive tasks developers face. Junior engineers spend 20–60 minutes per broken CI run tracing `pytest` failures back to their root cause. Senior engineers do it faster — but it still interrupts deep work constantly.

gREV is built on three observations:

**The task is genuinely hard for AI.** Reading a `pytest` traceback, identifying which file caused the failure, understanding whether it is a syntax error, logic error, or import mismatch — and then writing a correct fix — requires multi-step reasoning that current models still find challenging.

**The grader is perfectly deterministic.** `pytest` output is structured: `N passed, M failed`. Scoring requires no subjectivity. `score = passed / total`. This makes gREV trustworthy as an evaluation benchmark — you cannot accidentally inflate scores through prompt engineering the grader.

**The difficulty ladder is natural.** Syntax errors are visually obvious in a traceback. Logic errors require reading both the test's expectation and the source function. Multi-file import mismatches require tracing a dependency chain across files. Each tier requires a genuinely deeper level of reasoning.

---

## Architecture

```
gREV/
├── openenv.yaml              # OpenEnv manifest — spec compliance
├── inference.py              # Baseline agent (OpenAI client → HTTP API)
├── Dockerfile                # Multi-stage build → HF Spaces port 7860
├── pyproject.toml            # Project metadata and dependencies
│
├── grev/
│   ├── __init__.py
│   ├── models.py             # Pydantic v2: Observation, Action, Reward, StepResult
│   └── env.py                # Core state machine — subprocess sandbox, state reset
│
├── server/
│   └── app.py                # FastAPI — /reset /step /state /grade /health /tasks
│
└── tasks/
    ├── easy/                 # Broken repo: syntax error in main.py
    ├── medium/               # Broken repo: logic error in parser.py
    └── hard/                 # Broken repo: cross-file import mismatch
                              #   fetcher.py ↔ test_fetcher.py
```

### Core engine (`grev/env.py`)

The environment implements a safe subprocess execution model. Every `run_command` action is executed via `subprocess.run()` with a strict 15-second timeout — this prevents infinite loops or hanging processes from blocking the Hugging Face Space.

State isolation is enforced by copying the read-only task directory into `/tmp/workspace` on each `reset()`. At the start of a new episode, the previous workspace is wiped with `shutil.rmtree()` before a fresh copy is made with `shutil.copytree()`. There is no state bleed between episodes.

A background evaluator runs silently after every action. It calls `pytest` with `--tb=no -q` to get a fast pass/fail count without polluting the agent's stdout. This powers the **intermediate reward shaping** — the agent gets a signal after every step, not just at the end of the episode.

### Server (`server/app.py`)

FastAPI with a workspace-aware `/grade` endpoint that parses the final `pytest` output to produce a fractional score. The server maintains a single global environment instance per process, reinitialised on each `/reset` call. CORS is open for agent compatibility.

### Inference client (`inference.py`)

Dual-routed LLM script. Primary route: `Llama-3.3-70B-Versatile` via Groq (fast, free). Fallback: `meta-llama/Meta-Llama-3-8B-Instruct` via HF router. Uses the OpenAI Python client against both endpoints — zero code changes when switching providers.

The system prompt enforces a **"Senior Engineer" reasoning pattern**: the agent must read `pytest` output, inspect the relevant file, and reason about the bug before attempting any edit. This produces a measurable improvement in score over naive prompting.

---

## Observation space

Returned as JSON after every `reset()` and `step()` call.

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Active task: `"easy"` · `"medium"` · `"hard"` |
| `current_directory` | `str` | Absolute path to the agent's writable workspace |
| `directory_contents` | `list[str]` | All files present in the workspace root |
| `last_command_stdout` | `str` | Full stdout from the previous `run_command` action |
| `last_command_stderr` | `str` | Full stderr from the previous `run_command` action |
| `step` | `int` | Current step number (1-indexed) |
| `max_steps` | `int` | Total step budget for this episode (8) |
| `done` | `bool` | `true` when all tests pass or step budget is exhausted |
| `info` | `str` | Human-readable status hint |

---

## Action space

The agent sends a JSON object to `POST /step`. Two action types are supported.

### `run_command`

Execute any shell command inside the sandboxed workspace.

```json
{
  "action_type": "run_command",
  "command": "pytest tests/ -v"
}
```

The command runs from `current_directory` with a 15-second hard timeout. Stdout and stderr are captured and returned in the next observation. The agent can run any command that would be available to a developer: `pytest`, `cat`, `python -c`, `grep`, `ls`, `head`, etc.

### `edit_file`

Overwrite a workspace file with corrected content.

```json
{
  "action_type": "edit_file",
  "file_path": "src/main.py",
  "new_content": "def add(a, b):\n    return a + b\n"
}
```

The file is written atomically. Attempts to write outside the workspace root are rejected with a `-0.10` penalty and an error message in `info`.

---

## Tasks

### Task 1 · `easy` — syntax error

**Real-world analog:** A developer pushed a file with a syntax error — `pytest` fails immediately on collection with a `SyntaxError` before any test runs.

**What is broken:** `main.py` contains a deliberate syntax error — a missing colon after a `def` statement, an unclosed parenthesis, or a wrong indentation level that Python cannot parse.

**Agent strategy:**
1. Run `pytest` → see `SyntaxError` with file and line number
2. Run `cat main.py` → read the relevant lines
3. Run `edit_file` with the corrected content
4. Run `pytest` again to confirm

**Grader:** `pytest` exit 0 = 1.0. Partial credit: `passed / (passed + failed)`.

| Agent type | Expected score |
|-----------|---------------|
| Random | ~0.10 |
| GPT-4o-mini | ~0.75 |
| Llama-3.3-70B | ~0.85 |
| Perfect (1 step) | 1.0 + efficiency bonus |

---

### Task 2 · `medium` — logic error

**Real-world analog:** The code runs and imports cleanly, but returns wrong values. `pytest` assertions fail because the function produces incorrect output — not a crash, just wrong answers.

**What is broken:** A function in `parser.py` contains a logic error — an off-by-one index, a `+` where `-` is needed, or a reversed conditional branch. The file parses and imports successfully.

**Agent strategy:**
1. Run `pytest -v` → see which assertions fail, what values were expected vs actual
2. Run `cat parser.py` → read the function implementation
3. Identify the logic error by comparing expected output to code
4. Run `edit_file` with corrected logic
5. Run `pytest` to confirm

**Grader:** Fractional. Fixing the broken function typically fixes 2–4 tests at once.

| Agent type | Expected score |
|-----------|---------------|
| Random | ~0.08 |
| GPT-4o-mini | ~0.50 |
| Llama-3.3-70B | ~0.60 |
| Perfect | 1.0 + efficiency bonus |

---

### Task 3 · `hard` — cross-file import mismatch

**Real-world analog:** A refactor renamed a function or restructured a module's exports, but not all callers were updated. The traceback points to the caller, but the real fix is in the module that changed its interface.

**What is broken:** `fetcher.py` exports a function under a new name. `test_fetcher.py` imports the old name. The `ImportError` or `AttributeError` traceback points to the test file, but the correct fix may be in either file — the agent must reason about which side of the interface to change.

**Agent strategy:**
1. Run `pytest` → see `ImportError` or `AttributeError` with module path
2. Run `cat fetcher.py` → check what names are actually exported
3. Run `cat test_fetcher.py` → check what names are being imported
4. Decide: fix the export in `fetcher.py` or fix the import in `test_fetcher.py`
5. Apply the edit and confirm

This task genuinely challenges frontier models because the failure message blames the wrong file. The agent must read both sides of an interface and reason about which to change.

| Agent type | Expected score |
|-----------|---------------|
| Random | ~0.05 |
| GPT-4o-mini | ~0.30 |
| Llama-3.3-70B | ~0.35 |
| Perfect | 1.0 + efficiency bonus |

---

## Reward function

### Per-step reward (intermediate signal)

The background evaluator runs `pytest --tb=no -q` silently after every action and computes a delta signal — the change in test pass rate since the previous step.

| Outcome | Step reward |
|---------|------------|
| Tests improved (more passing than before) | `+0.10` |
| Tests unchanged | `0.00` |
| Tests regressed (edit made things worse) | `-0.05` |
| `run_command` timed out (15s exceeded) | `-0.05` |
| `edit_file` path outside workspace | `-0.10` |
| `edit_file` wrote unparseable Python | `-0.03` |

### Episode reward (`/grade`)

Computed deterministically from the final `pytest` output at episode end:

```python
# Parse: "3 passed, 2 failed in 0.45s"
score = passed_count / (passed_count + failed_count)

# Efficiency bonus: reward finishing early when all tests pass
steps_remaining = max_steps - steps_taken
efficiency_bonus = 0.10 * (steps_remaining / max_steps) if score == 1.0 else 0.0

total_reward = min(1.0, score + efficiency_bonus)
```

A perfect solve in 2 steps (out of 8) scores `1.0 + 0.10 × (6/8) = 1.075`, clamped to `1.0`. This incentivises agents to diagnose efficiently rather than exhausting their step budget.

---

## Setup

### Local development

```bash
git clone https://github.com/LangerSword/gREV
cd gREV

# with uv (recommended)
uv sync

# or with pip
pip install -e .

# start the server
cd server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

Verify:
```bash
curl http://localhost:7860/health
# → {"status": "ok", "tasks": ["easy", "medium", "hard"]}
```

### Docker

```bash
docker build -t grev-env .
docker run -p 7860:7860 grev-env
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | Hugging Face token (used as API key for LLM routing) |
| `API_KEY` | — | Fallback if `HF_TOKEN` not set |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API base URL |
| `MODEL_NAME` | `meta-llama/Llama-3.3-70B-Instruct` | Model for the baseline agent |
| `ENV_URL` | `http://localhost:7860` | Environment server URL for inference script |

---

## Running the baseline agent

The inference script uses the OpenAI Python client against the HF router (free with any HF token). It calls the environment exclusively over HTTP — no internal imports.

```bash
export HF_TOKEN=hf_your_token_here
export ENV_URL=https://langersword-grev-openenv.hf.space

python inference.py
```

For local testing:
```bash
export ENV_URL=http://localhost:7860
python inference.py
```

To use Groq (faster, also free):
```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export API_KEY=gsk_your_groq_key
export MODEL_NAME=llama-3.3-70b-versatile
python inference.py
```

Expected output:
```
========================================
Task: easy
========================================
  Step  1: run_command      | pytest -v       | r=+0.000 | cum=+0.000
  Step  2: run_command      | cat main.py     | r=+0.000 | cum=+0.000
  Step  3: edit_file        | src/main.py     | r=+0.100 | cum=+0.100
  Step  4: run_command      | pytest -v       | r=+0.100 | cum=+0.200
  Done at step 4
  Final score: 1.000  ████████████████████████

========================================
BASELINE RESULTS
========================================
  easy                        1.000  ████████████████████████
  medium                      1.000  ████████████████████████
  hard                        1.000  ████████████████████████
  AVERAGE                     1.000
========================================
```

---

## API reference

### `GET /health`

Liveness check used by the OpenEnv validation script.

```json
{"status": "ok", "tasks": ["easy", "medium", "hard"]}
```

---

### `GET /tasks`

List all available tasks with metadata.

```json
[
  {
    "task_id": "easy",
    "description": "Fix a syntax error in main.py",
    "difficulty": "easy",
    "max_steps": 8,
    "reward_range": [0.0, 1.0]
  },
  {
    "task_id": "medium",
    "description": "Fix a logic error in parser.py",
    "difficulty": "medium",
    "max_steps": 8,
    "reward_range": [0.0, 1.0]
  },
  {
    "task_id": "hard",
    "description": "Fix a cross-file import mismatch between fetcher.py and test_fetcher.py",
    "difficulty": "hard",
    "max_steps": 8,
    "reward_range": [0.0, 1.0]
  }
]
```

---

### `POST /reset`

Start a new episode. Wipes the previous workspace and copies a fresh broken repo into `/tmp/workspace`.

**Request body:**
```json
{
  "task_id": "easy",
  "seed": 42
}
```

**Response:** `Observation` object

```json
{
  "task_id": "easy",
  "current_directory": "/tmp/workspace",
  "directory_contents": ["main.py", "tests/", "tests/test_main.py"],
  "last_command_stdout": "",
  "last_command_stderr": "",
  "step": 0,
  "max_steps": 8,
  "done": false,
  "info": "Episode started. Workspace is ready. Run pytest to see what is broken."
}
```

---

### `POST /step`

Take one action. Returns the updated observation, per-step reward, and done flag.

**Request body (run_command):**
```json
{
  "action_type": "run_command",
  "command": "pytest tests/ -v"
}
```

**Request body (edit_file):**
```json
{
  "action_type": "edit_file",
  "file_path": "main.py",
  "new_content": "def greet(name: str) -> str:\n    return f'Hello, {name}'\n"
}
```

**Response:** `StepResult` object

```json
{
  "observation": { "...": "updated Observation" },
  "reward": {
    "total": 0.10,
    "breakdown": {"test_improvement": 0.10},
    "message": "Tests improved: 3 passing → 5 passing"
  },
  "done": false,
  "info": {}
}
```

---

### `POST /grade`

Compute and return the final episode score. Call this after the episode ends (`done: true`) or at any point to check current standing.

**Response:** `EpisodeResult` object

```json
{
  "task_id": "easy",
  "total_reward": 1.0,
  "steps_taken": 4,
  "success": true,
  "breakdown": {
    "pytest_score": 0.95,
    "efficiency_bonus": 0.05
  }
}
```

`success` is `true` when `total_reward >= 0.70`.

---

### `GET /state`

Return the full internal environment state. Useful for debugging agent behaviour or inspecting mid-episode state without affecting it.

```json
{
  "task_id": "easy",
  "seed": 42,
  "step": 3,
  "initialized": true,
  "workspace": "/tmp/workspace",
  "last_pytest_passed": 4,
  "last_pytest_total": 5
}
```

---

## Baseline scores

Achieved with `Llama-3.3-70B-Versatile` via Groq, seed 42, all three tasks:

| Task | Score | Steps used | Time to solve |
|------|-------|------------|---------------|
| easy | **1.000** | 4 / 8 | ~12s |
| medium | **1.000** | 5 / 8 | ~18s |
| hard | **1.000** | 6 / 8 | ~22s |
| **average** | **1.000** | | |

The "Senior Engineer" system prompt — which forces the agent to read pytest output and inspect files before attempting edits — is key to achieving consistent 1.0 scores. Without it, the same model scores approximately 0.55 on the hard task.

---

## Validation

Install the OpenEnv CLI and run the validation suite:

```bash
pip install openenv-core
openenv validate
```

Three checks run in sequence:

1. **HF Space ping** — `POST https://langersword-grev-openenv.hf.space/reset` must return HTTP 200
2. **Docker build** — `docker build` from the repo root must exit 0
3. **Schema validation** — `openenv.yaml` must be valid and all model paths must resolve

All three pass. ✓

---

## License

MIT — see [LICENSE](LICENSE)

---

*Built for the Scaler × Meta / Hugging Face OpenEnv Hackathon, April 2026.*
