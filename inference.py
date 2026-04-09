import asyncio
import json
import os
import re
import sys
import textwrap
from typing import Any, List, Optional

import requests
from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "gREV"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = 0.1
MAX_TOKENS = 700


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a coding agent inside a broken Python repository.
    Return exactly one JSON object per turn.
    Allowed actions:
    1) {"action_type":"run_command","command":"..."}
    2) {"action_type":"edit_file","file_path":"...","new_content":"..."}
    Keep responses valid JSON only.
    """
).strip()


def _extract_json_obj(text: str) -> Optional[dict[str, Any]]:
    cleaned = re.sub(r"```(?:json)?", "", text or "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception as exc:
        print(f"[DEBUG] JSON parse (direct) failed: {exc}", flush=True)

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception as exc:
        print(f"[DEBUG] JSON parse (regex block) failed: {exc}", flush=True)
        return None


def _safe_json_from_response(response: requests.Response) -> Optional[dict[str, Any]]:
    try:
        payload = response.json()
    except Exception as exc:
        print(f"[DEBUG] Response JSON parsing failed: {exc}", flush=True)
        return None
    if isinstance(payload, dict):
        return payload
    print("[DEBUG] Response JSON is not an object", flush=True)
    return None


def _post_reset(task_id: str) -> Optional[dict[str, Any]]:
    try:
        response = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        print(f"[DEBUG] /reset request failed for task {task_id}: {exc}", flush=True)
        return None

    payload = _safe_json_from_response(response)
    if payload is None:
        return None
    observation = payload.get("observation")
    if isinstance(observation, dict):
        return observation
    print(f"[DEBUG] /reset missing observation for task {task_id}", flush=True)
    return None


def _post_step(action: dict[str, Any]) -> Optional[dict[str, Any]]:
    try:
        response = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        print(f"[DEBUG] /step request failed: {exc}", flush=True)
        return None

    payload = _safe_json_from_response(response)
    return payload


def _post_grade(task_id: str) -> Optional[dict[str, Any]]:
    try:
        response = requests.post(f"{ENV_URL}/grade", json={"task_id": task_id}, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        print(f"[DEBUG] /grade request failed for task {task_id}: {exc}", flush=True)
        return None

    payload = _safe_json_from_response(response)
    return payload


def _llm_action(client: OpenAI, messages: List[dict[str, str]]) -> Optional[dict[str, Any]]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return None

    raw_text = ""
    try:
        raw_text = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM response parsing failed: {exc}", flush=True)
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


def _build_user_prompt(observation: dict[str, Any], step: int) -> str:
    current_directory = observation.get("current_directory", "")
    directory_contents = observation.get("directory_contents", [])
    last_stdout = observation.get("last_command_stdout", "")
    last_stderr = observation.get("last_command_stderr", "")
    done_val = observation.get("done", False)

    return textwrap.dedent(
        f"""
        Step: {step}/{MAX_STEPS}
        current_directory: {current_directory}
        directory_contents: {directory_contents}
        last_command_stdout:
        {last_stdout}

        last_command_stderr:
        {last_stderr}

        done: {done_val}
        Return only one valid JSON action.
        """
    ).strip()


def _extract_step_error(step_payload: dict[str, Any], observation: dict[str, Any]) -> Optional[str]:
    info = step_payload.get("info")
    if isinstance(info, dict):
        candidate = info.get("error") or info.get("last_action_error")
        if candidate:
            return str(candidate)

    obs_info = observation.get("info")
    if isinstance(obs_info, dict):
        candidate = obs_info.get("error") or obs_info.get("last_action_error")
        if candidate:
            return str(candidate)

    return None


def run_task(client: OpenAI, task_name: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = _post_reset(task_name)
        if observation is None:
            print(f"[DEBUG] Task {task_name}: reset failed, ending task", flush=True)
            return

        messages: List[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            steps_taken = step

            messages.append({"role": "user", "content": _build_user_prompt(observation, step)})
            action = _llm_action(client, messages)

            if action is None:
                print(f"[DEBUG] Task {task_name}: LLM action unavailable, ending task", flush=True)
                break

            try:
                action_str = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
            except Exception as exc:
                print(f"[DEBUG] Action serialization failed: {exc}", flush=True)
                break

            messages.append({"role": "assistant", "content": action_str})

            step_payload = _post_step(action)
            if step_payload is None:
                print(f"[DEBUG] Task {task_name}: step request failed, ending task", flush=True)
                break

            try:
                observation_value = step_payload.get("observation", {})
                observation = observation_value if isinstance(observation_value, dict) else {}
            except Exception as exc:
                print(f"[DEBUG] Step observation parsing failed: {exc}", flush=True)
                break

            try:
                reward_raw = step_payload.get("reward", 0.0)
                reward = float(reward_raw)
            except Exception as exc:
                print(f"[DEBUG] Step reward parsing failed: {exc}", flush=True)
                reward = 0.0

            try:
                done = bool(step_payload.get("done", False))
            except Exception as exc:
                print(f"[DEBUG] Step done parsing failed: {exc}", flush=True)
                done = False

            error = _extract_step_error(step_payload, observation)
            rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        grade_payload = _post_grade(task_name)
        if grade_payload is None:
            print(f"[DEBUG] Task {task_name}: grade failed, using fallback score", flush=True)
        else:
            try:
                score = float(grade_payload.get("total_reward", 0.0))
            except Exception as exc:
                print(f"[DEBUG] Grade score parsing failed: {exc}", flush=True)
                score = 0.0

        score = max(0.0, min(1.0, score))
        success = score > 0.0

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} crashed with unhandled path: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    if not API_KEY:
        print("[DEBUG] Missing API key: set GROQ_API_KEY or HF_TOKEN", flush=True)
        for task_name in TASKS:
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[DEBUG] Failed to create OpenAI client: {exc}", flush=True)
        for task_name in TASKS:
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    for task_name in TASKS:
        run_task(client, task_name)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[DEBUG] Global failure: {exc}", flush=True)
        sys.exit(0)
    sys.exit(0)

