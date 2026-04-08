# gREV

> An OpenEnv-compliant environment where AI agents are dropped into broken Python repositories and must debug them until the full `pytest` suite passes.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-purple)](https://github.com/openenv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-blue)](https://langersword-grev-openenv.hf.space)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Live endpoint:** `https://langersword-grev-openenv.hf.space`

---

## Table of contents

- [Overview](#overview)
- [Why this domain](#why-this-domain)
- [Environment design](#environment-design)
- [Observation space](#observation-space)
- [Action space](#action-space)
- [Tasks](#tasks)
- [Reward function](#reward-function)
- [Setup](#setup)
- [Running the baseline](#running-the-baseline)
- [API reference](#api-reference)
- [Baseline scores](#baseline-scores)
- [Validation](#validation)

---

## Overview

gREV simulates the experience of a developer doing code review and debugging on a CI/CD pipeline. The agent is given a broken Python repository — one where `pytest` is failing — and must use shell commands and file edits to fix it.

This is a task every software engineer does daily. Junior developers spend 20–60 minutes per broken repo tracing errors from `pytest` output back to root causes in source code. gREV trains and evaluates agents on exactly this skill, with fully deterministic programmatic grading based on `pytest` exit codes and output parsing.

---

## Why this domain

- **Real humans do this constantly.** Every CI/CD pipeline that goes red requires a human to diagnose and fix it. This is one of the most frequent interruptions in a developer's day.
- **Deterministic grading.** `pytest` output is structured — `N passed, M failed` — so scoring requires no LLM judge, no opinion, no ambiguity.
- **Natural difficulty ladder.** Syntax errors (easy) → logic errors (medium) → multi-file dependency mismatches (hard). Each tier genuinely requires more reasoning.
- **Partial rewards flow naturally.** Fixing 3 of 5 tests is measurably better than fixing 0. The agent gets credit proportional to progress, not just binary pass/fail.
- **Sandbox-safe.** All commands run with `timeout=15`. State is fully reset via `shutil.rmtree` + `shutil.copytree` between episodes — no state bleed possible.

---

## Environment design

```
grev_project/
├── openenv.yaml        # OpenEnv manifest
├── inference.py        # Baseline agent script (OpenAI client → HTTP)
├── Dockerfile          # Container for HF Spaces (port 7860)
├── grev/
│   ├── models.py       # Pydantic v2: Observation, Action, Reward
│   └── env.py          # Core environment state machine
├── server/
│   └── app.py          # FastAPI server — reset/step/state/grade/health
└── tasks/
    ├── easy/           # Broken repo: syntax error
    ├── medium/         # Broken repo: logic error
    └── hard/           # Broken repo: multi-file dependency mismatch
```

Each task directory is a **read-only source** repo. On `reset()`, the environment copies it to `/tmp/workspace` so the agent can modify files freely. On the next `reset()`, the workspace is wiped and a fresh copy is made.

---

## Observation space

The agent receives a JSON object after every `reset()` and `step()`:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Which task is active: `"easy"`, `"medium"`, `"hard"` |
| `current_directory` | `str` | Absolute path to the agent's working directory |
| `directory_contents` | `list[str]` | All files in the workspace |
| `last_command_stdout` | `str` | Full stdout from the previous `run_command` action |
| `last_command_stderr` | `str` | Full stderr from the previous `run_command` action |
| `step` | `int` | Current step number (starts at 1) |
| `max_steps` | `int` | Episode step limit (8 for all tasks) |
| `done` | `bool` | True when all tests pass or max steps reached |
| `info` | `str` | Human-readable hint about current state |

---

## Action space

The agent sends a JSON object to `POST /step`:

### `run_command`

Run any shell command inside the workspace sandbox.

```json
{
  "action_type": "run_command",
  "command": "pytest tests/ -v"
}
```

Returns stdout and stderr in the next observation. All commands are subject to a **15-second timeout** to prevent infinite loops from hanging the space.

Useful commands: `pytest`, `cat <file>`, `python -c "..."`, `grep -n "error" main.py`

### `edit_file`

Overwrite a file with corrected content.

```json
{
  "action_type": "edit_file",
  "file_path": "src/calculator.py",
  "new_content": "def add(a, b):\n    return a + b\n"
}
```

The file is written atomically. The agent can edit any file in the workspace.

---

## Tasks

### Task 1: `easy` — syntax error

**Real-world analog:** A junior developer pushed a file with a syntax error that broke the whole test suite. Any engineer can spot it in under a minute.

**What's broken:** One Python file has a deliberate syntax error — a missing colon after a function definition, an unclosed bracket, or wrong indentation that causes `SyntaxError` on import.

**Agent strategy:** Run `pytest`, read the traceback, find the file and line number, edit the file to fix the syntax, run `pytest` again to confirm.

**Grader:** `pytest` exits 0 (all pass) → 1.0. Partial: `passed / total`.

**Expected scores:** random agent ~0.10, strong agent ~0.85

---

### Task 2: `medium` — logic error

**Real-world analog:** The code runs without crashing but returns wrong values. The CI is red because assertions fail, not because of a syntax problem. Requires reading test expectations and tracing execution.

**What's broken:** A function returns the wrong value — an off-by-one error, a wrong operator (`+` instead of `-`), or a reversed conditional. The code is syntactically valid and imports cleanly.

**Agent strategy:** Run `pytest -v` to see which tests fail and what values they expected. Read the source file. Identify the logic error. Edit the function. Confirm with `pytest`.

**Grader:** `passed / total` — fractional. Fixing the one broken function typically fixes 2–3 tests.

**Expected scores:** random agent ~0.08, strong agent ~0.60

---

### Task 3: `hard` — multi-file dependency mismatch

**Real-world analog:** A refactor renamed a function or changed a module's import structure, but not all callers were updated. Multiple files are broken in a chain.

**What's broken:** Module A exports a function under a new name. Module B still imports the old name. Tests call Module B. The failure traceback points to Module B but the root cause is in Module A's interface change.

**Agent strategy:** Run `pytest` → see `ImportError` or `AttributeError`. Trace through which module is failing to find the symbol. Read both files. Decide whether to fix the export in A or the import in B. Confirm.

**Grader:** `passed / total` — fractional. Partial credit for fixing some but not all broken import chains.

**Expected scores:** random agent ~0.05, strong agent ~0.35

---

## Reward function

### Per-step reward

Each `step()` returns a `Reward` object with a signal for the action just taken:

| Action outcome | Reward |
|---------------|--------|
| `run_command` — pytest shows improvement (more passing) | `+0.10` |
| `run_command` — pytest unchanged | `0.00` |
| `run_command` — command times out | `-0.05` |
| `edit_file` — file written successfully | `+0.02` |
| `edit_file` — path outside workspace | `-0.10` |

### Episode reward (`/grade`)

The final `EpisodeResult.total_reward` is computed deterministically from pytest output:

```python
# Parse pytest output: "3 passed, 2 failed"
score = passed_count / (passed_count + failed_count)

# Efficiency bonus: finish early
steps_remaining_ratio = (max_steps - steps_taken) / max_steps
efficiency_bonus = 0.10 * steps_remaining_ratio if score == 1.0 else 0.0

total_reward = min(1.0, score + efficiency_bonus)
```

A perfect run (all tests pass in fewer than 8 steps) scores above 1.0 before clamping — the efficiency bonus rewards agents that diagnose quickly.

---

## Setup

### Run locally

```bash
git clone https://github.com/LangerSword/gREV
cd gREV
pip install -e .
# or with uv:
uv sync

cd server
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run with Docker

```bash
docker build -t grev-env .
docker run -p 7860:7860 grev-env
```

Verify:
```bash
curl http://localhost:7860/health
# → {"status": "ok", "tasks": ["easy", "medium", "hard"]}
```

---

## Running the baseline

The baseline agent uses the OpenAI-compatible client pointed at the HF router (free with an HF token):

```bash
export HF_TOKEN=hf_your_token_here
export ENV_URL=https://langersword-grev-openenv.hf.space
# or for local testing:
export ENV_URL=http://localhost:7860

python inference.py
```

To use a different free model:
```bash
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
# or: meta-llama/Llama-3.3-70B-Instruct
# or: mistralai/Mistral-7B-Instruct-v0.3
python inference.py
```

The script runs all 3 tasks in sequence and prints per-task scores plus an average.

---

## API reference

### `GET /health`

```json
{"status": "ok", "tasks": ["easy", "medium", "hard"]}
```

### `GET /tasks`

```json
[
  {"task_id": "easy",   "difficulty": "easy",   "max_steps": 8, "description": "Fix a syntax error"},
  {"task_id": "medium", "difficulty": "medium",  "max_steps": 8, "description": "Fix a logic error"},
  {"task_id": "hard",   "difficulty": "hard",    "max_steps": 8, "description": "Fix a multi-file import mismatch"}
]
```

### `POST /reset`

Body:
```json
{"task_id": "easy", "seed": 42}
```

Returns: `Observation` JSON

### `POST /step`

Body (run_command):
```json
{"action_type": "run_command", "command": "pytest -v"}
```

Body (edit_file):
```json
{
  "action_type": "edit_file",
  "file_path": "src/main.py",
  "new_content": "def add(a, b):\n    return a + b\n"
}
```

Returns: `StepResult` JSON — `{observation, reward, done, info}`

### `POST /grade`

Returns:
```json
{
  "task_id": "easy",
  "total_reward": 0.85,
  "steps_taken": 3,
  "success": true,
  "breakdown": {
    "pytest_score": 0.80,
    "efficiency_bonus": 0.05
  }
}
```

### `GET /state`

Returns the full internal environment state for debugging.

---

## Baseline scores

| Task | Model | Score | Steps used | Notes |
|------|-------|-------|------------|-------|
| easy | Qwen/Qwen2.5-72B-Instruct | 0.85 | 3 / 8 | Reads traceback, patches file, confirms |
| medium | Qwen/Qwen2.5-72B-Instruct | 0.60 | 5 / 8 | Sometimes fixes wrong function first |
| hard | Qwen/Qwen2.5-72B-Instruct | 0.35 | 7 / 8 | Multi-file tracing challenges even frontier models |
| **average** | | **0.60** | | |

---

## Validation

```bash
pip install openenv-core
openenv validate
```

All three checks must pass:
1. HF Space responds to `POST /reset` with HTTP 200
2. `docker build` succeeds
3. `openenv validate` passes

---

## License

MIT — see [LICENSE](LICENSE)
