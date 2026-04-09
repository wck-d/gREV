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
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""

BENCHMARK = "grev"
MAX_STEPS = 8
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

    action_type = parsed.get("action_type")
    if action_type not in {"run_command", "edit_file"}:
        print(f"[DEBUG] Invalid action_type from LLM: {action_type}", flush=True)
        return None
    return parsed


def _build_user_prompt(obs, step: int) -> str:
    """Build the user prompt from the observation."""
    return textwrap.dedent(
        f"""
        Step: {step}/{MAX_STEPS}
        current_directory: {obs.current_directory}
        directory_contents: {obs.directory_contents}
        last_command_stdout:
        {obs.last_command_stdout}

        last_command_stderr:
        {obs.last_command_stderr}

        done: {obs.done}
        Return only one valid JSON action.
        """
    ).strip()


def _deterministic_action(obs, step: int, task: str) -> Optional[dict]:
    """Rule-based fallback when LLM is unavailable."""
    if step == 1:
        return {"action_type": "run_command", "command": "pytest"}

    stdout = obs.last_command_stdout or ""

    # After pytest, cat the broken file(s) to understand the problem
    if step == 2:
        if task == "easy":
            return {"action_type": "run_command", "command": "cat calculator.py"}
        elif task == "medium":
            return {"action_type": "run_command", "command": "cat data_processor.py"}
        elif task == "hard":
            return {"action_type": "run_command", "command": "cat auth.py"}

    # For hard task, also read models.py on step 3
    if step == 3 and task == "hard":
        return {"action_type": "run_command", "command": "cat models.py"}

    return None


def _run_episode(task: str, seed: int) -> float:
    """Run a single task episode."""
    # Required by submission rules: create OpenAI client
    client = _build_llm_client()
    use_llm = bool(HF_TOKEN)

    env = gREVEnv()
    obs = env.reset(task_level=task, seed=seed)

    rewards: list[float] = []
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    for step in range(1, MAX_STEPS + 1):
        if obs.done:
            break

        # Get action from LLM or fallback
        action_dict = None
        if use_llm:
            messages.append({"role": "user", "content": _build_user_prompt(obs, step)})
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
        return ["easy", "medium", "hard"]
    return [task_arg]


def main():
    parser = argparse.ArgumentParser(description="gREV baseline inference")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
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
