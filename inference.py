import os
import json
import requests
import re
import time
from openai import OpenAI

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# Fix malformed URL (just in case)
if ENV_URL.startswith("[") and "](" in ENV_URL:
    match = re.search(r'\((https?://.*?)\)', ENV_URL)
    if match:
        ENV_URL = match.group(1)

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
GROQ_MODEL = "llama-3.3-70b-versatile"


# ✅ SAFE LLM FUNCTION (never crashes)
def get_llm_action(messages):
    if GROQ_API_KEY:
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[!] Groq failed: {e}")

    if HF_TOKEN:
        try:
            response = hf_client.chat.completions.create(
                model=HF_MODEL,
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[!] HF failed: {e}")

    # ✅ fallback (NO crash)
    print("[!] No API keys → fallback")
    return '{"action_type": "run_command", "command": "pytest"}'


def run_inference():
    print("Waiting for server...")
    time.sleep(5)  # ✅ VERY IMPORTANT

    tasks = ["easy", "medium", "hard"]
    max_steps = 10

    for task in tasks:
        print(f"\n[START] Task: {task}")

        # RESET
        try:
            reset_res = requests.post(
                f"{ENV_URL}/reset",
                json={"task_id": task},
                timeout=10
            )
            reset_res.raise_for_status()
            obs = reset_res.json()
        except Exception as e:
            print(f"[!] Reset failed: {e}")
            return  # stop instead of crashing later

        messages = [
            {
                "role": "system",
                "content": "You are a Python debugging agent. Output ONLY JSON."
            }
        ]

        for step in range(max_steps):
            print(f"[STEP] {step}")

            messages.append({
                "role": "user",
                "content": f"Observation: {json.dumps(obs)}"
            })

            try:
                # safe overrides
                if step == 0:
                    action_payload = {"action_type": "run_command", "command": "pytest"}
                elif step == 1:
                    action_payload = {"action_type": "run_command", "command": "ls"}
                else:
                    raw_action = get_llm_action(messages)

                    if not raw_action:
                        raise ValueError("Empty LLM output")

                    if raw_action.startswith("```json"):
                        raw_action = raw_action[7:]
                    if raw_action.startswith("```"):
                        raw_action = raw_action[3:]
                    if raw_action.endswith("```"):
                        raw_action = raw_action[:-3]

                    action_payload = json.loads(raw_action.strip())

                print(f"Action: {action_payload}")

                messages.append({
                    "role": "assistant",
                    "content": json.dumps(action_payload)
                })

            except Exception as e:
                print(f"[!] LLM error: {e} → fallback")
                action_payload = {"action_type": "run_command", "command": "pytest"}

            # STEP
            try:
                step_res = requests.post(
                    f"{ENV_URL}/step",
                    json={"action": action_payload},
                    timeout=10
                )
                step_res.raise_for_status()
                obs = step_res.json()
            except Exception as e:
                print(f"[!] Step failed: {e}")
                break

            if obs.get("done"):
                print("[✓] Task complete")
                break

        # GRADE
        try:
            grade_res = requests.post(
                f"{ENV_URL}/grade",
                json={"task_id": task},
                timeout=10
            )
            grade_res.raise_for_status()
            grade_data = grade_res.json()

            print(f"[END] {task} → {grade_data}")

        except Exception as e:
            print(f"[!] Grade failed: {e}")


if __name__ == "__main__":
    try:
        print("Starting inference...")
        run_inference()
    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
