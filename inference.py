"""Reproducible baseline inference for gREV (RepoRescueEnv).

Usage:
  python inference.py --task all --episodes 1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from typing import Iterable, List, Optional

from openai import OpenAI

try:
    from grev.models import GrevAction
    from grev.env import gREVEnv
except ImportError:
    from models import GrevAction
    from grev.env import gREVEnv


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
# Rubric requires reading from OPENAI_API_KEY — support all common env var names
HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
    or ""
)

BENCHMARK = "grev"
# Note: per-episode step budget comes from TASK_CONFIGS, not this constant
DEFAULT_MAX_STEPS = 24  # upper safety ceiling only
TEMPERATURE = 0.1
MAX_TOKENS = 700

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a coding agent inside a broken Python repository.
    Your goal is to diagnose and fix the code so all pytest tests pass.

    You have two actions available:
    1) {"action_type":"run_command","command":"<shell command>"}
    2) {"action_type":"edit_file","file_path":"<path>","new_content":"<full file content>"}

    Strategy:
    - First, run "pytest" to see what's failing.
    - Then "cat <filename>" to read the broken file.
    - Then use edit_file to fix it.
    - Finally run "pytest" again to verify.

    Return exactly one JSON object per turn. No extra text.
    """
).strip()


def _build_llm_client() -> OpenAI:
    """Create the mandatory OpenAI-compatible client from env configuration."""
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={task} env={env} model={model} "
        f"api_base={API_BASE_URL} hf_token={'set' if HF_TOKEN else 'unset'}"
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _extract_json_obj(text: str) -> Optional[dict]:
    """Try to extract a JSON object from LLM output."""
    cleaned = re.sub(r"```(?:json)?", "", text or "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


def _normalize_action(parsed: dict) -> Optional[dict]:
    """Map common LLM mistakes to valid action types.

    LLMs sometimes emit action_type='cat' or 'grep' etc. Normalize these
    to run_command so episodes don't silently skip steps.
    """
    action_type = parsed.get("action_type", "")
    if action_type in {"run_command", "edit_file"}:
        return parsed

    # Common LLM mistakes: command-style action types
    shell_verbs = {"cat", "grep", "ls", "head", "tail", "pytest", "python",
                   "run", "execute", "bash", "shell", "command"}
    if action_type.lower() in shell_verbs:
        command = parsed.get("command") or parsed.get("cmd") or action_type
        print(f"[DEBUG] Normalized action_type={action_type!r} to run_command", flush=True)
        return {"action_type": "run_command", "command": command}

    print(f"[DEBUG] Invalid action_type from LLM: {action_type}", flush=True)
    return None


def _llm_action(client: OpenAI, messages: list) -> Optional[dict]:
    """Ask the LLM for the next action."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return None

    parsed = _extract_json_obj(raw_text)
    if not isinstance(parsed, dict):
        print("[DEBUG] LLM output did not contain valid JSON action", flush=True)
        return None

    return _normalize_action(parsed)



def _build_user_prompt(obs, step: int, max_steps: int) -> str:
    """Build the user prompt from the observation."""
    return textwrap.dedent(
        f"""
        Step: {step}/{max_steps}
        current_directory: {obs.current_directory}
        directory_contents: {obs.directory_contents}
        last_command_stdout:
        {obs.last_command_stdout}

        last_command_stderr:
        {obs.last_command_stderr}

        reward_so_far: {obs.reward:.3f}
        done: {obs.done}
        Return only one valid JSON action.
        """
    ).strip()


def _deterministic_action(obs, step: int, task: str) -> Optional[dict]:
    """Rule-based fallback when LLM is unavailable or fails to produce JSON.

    Never returns None after step 1 — always keeps the episode alive by
    re-running pytest so the agent continues getting reward signal.
    """
    stdout = obs.last_command_stdout or ""

    if step == 1:
        return {"action_type": "run_command", "command": "pytest -v"}

    # After pytest, read the main broken file
    task_files = {
        "easy": "calculator.py",
        "medium": "data_processor.py",
        "hard": "auth.py",
        "medium_hard": "pipeline.py",
        "very_hard": "storage.py",
    }
    if step == 2:
        return {"action_type": "run_command", "command": f"cat {task_files.get(task, 'main.py')}"}

    # For cross-file tasks, read the second file on step 3
    if step == 3 and task == "hard":
        return {"action_type": "run_command", "command": "cat models.py"}
    if step == 3 and task == "very_hard":
        return {"action_type": "run_command", "command": "cat test_storage.py"}

    # Keep running pytest to track progress — never return None
    return {"action_type": "run_command", "command": "pytest -v"}


def _run_episode(task: str, seed: int) -> float:
    """Run a single task episode."""
    from grev.env import TASK_CONFIGS  # use per-task step budget

    client = _build_llm_client()
    use_llm = bool(HF_TOKEN)

    env = gREVEnv()
    obs = env.reset(task_level=task, seed=seed)

    # Use the task's configured step budget, not a hardcoded constant
    task_max_steps = TASK_CONFIGS.get(task, TASK_CONFIGS["easy"]).max_steps

    rewards: list[float] = []
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    for step in range(1, task_max_steps + 1):
        if obs.done:
            break

        # Get action from LLM or fallback
        action_dict = None
        if use_llm:
            messages.append({"role": "user", "content": _build_user_prompt(obs, step, task_max_steps)})
            action_dict = _llm_action(client, messages)

        if action_dict is None:
            action_dict = _deterministic_action(obs, step, task)

        if action_dict is None:
            break

        # Build typed action
        action = GrevAction(
            action_type=action_dict["action_type"],
            command=action_dict.get("command"),
            file_path=action_dict.get("file_path"),
            new_content=action_dict.get("new_content"),
        )

        obs = env.step(action)

        reward = float(obs.reward or 0.0)
        rewards.append(reward)

        action_str = action_dict.get("command") or action_dict.get("action_type", "unknown")
        error_text = obs.last_error if obs.last_error else None

        log_step(step=step, action=action_str, reward=reward, done=obs.done, error=error_text)

        if use_llm and action_dict:
            try:
                messages.append(
                    {"role": "assistant", "content": json.dumps(action_dict, ensure_ascii=False)}
                )
            except Exception:
                pass

        if obs.done:
            break

    # Final grading
    final_score, _ = env.grade()
    final_score = max(0.0, min(1.0, final_score))
    success = final_score >= 0.5

    rewards_csv_vals = rewards if rewards else [0.0]
    log_end(
        success=success,
        steps=env.state.step_count,
        score=final_score,
        rewards=rewards_csv_vals,
    )

    env.close()
    return final_score


def _task_list(task_arg: str) -> Iterable[str]:
    if task_arg == "all":
        return ["easy", "medium", "hard", "medium_hard", "very_hard"]
    return [task_arg]


def main():
    parser = argparse.ArgumentParser(description="gREV baseline inference")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "medium_hard", "very_hard", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary_scores: list[float] = []
    for task in _task_list(args.task):
        for episode_index in range(args.episodes):
            episode_seed = args.seed + (episode_index * 17)
            score = _run_episode(task=task, seed=episode_seed)
            summary_scores.append(score)

    if summary_scores:
        avg = sum(summary_scores) / len(summary_scores)
        print(f"[SUMMARY] episodes={len(summary_scores)} avg_score={avg:.3f}")


if __name__ == "__main__":
    main()
