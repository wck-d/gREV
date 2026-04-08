import os
import json
import requests
import re
from openai import OpenAI

# --- Configuration & Environment Variables ---
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# Quick sanitize to strip Markdown if it accidentally gets passed into the terminal export
if ENV_URL.startswith("[") and "](" in ENV_URL:
    match = re.search(r'\((https?://.*?)\)', ENV_URL)
    if match:
        ENV_URL = match.group(1)

# API Keys (Loaded cleanly from environment, no hardcoding!)
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Primary: Hugging Face (Llama 3 8B)
hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)
HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Fallback: Groq (Llama 3.1 8B - Blazing fast free tier)
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)
# The Smart Model (Llama 3.3 70B is a genius at coding and free on Groq)
GROQ_MODEL = "llama-3.3-70b-versatile"

def get_llm_action(messages):
    """Forces Groq's 70B model first for maximum intelligence."""
    # 1. ALWAYS Try Groq 70B First!
    if GROQ_API_KEY:
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1 # Low temperature so it doesn't hallucinate
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"      [!] Groq API failed: {e}. Trying Hugging Face...")
    
    # 2. Fallback to HF 8B only if Groq crashes
    if HF_TOKEN:
        try:
            response = hf_client.chat.completions.create(
                model=HF_MODEL,
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"      [!] Primary HF API also failed ({e}).")
            
    raise Exception("No valid API keys found in environment or both APIs crashed.")

def run_inference():
    tasks = ["easy", "medium", "hard"]
    max_steps = 10

    for task in tasks:
        print(f"\n[START] Task: {task}")

        # 1. Reset the environment via HTTP
        try:
            reset_res = requests.post(f"{ENV_URL}/reset", json={"task_id": task})
            reset_res.raise_for_status()
            obs = reset_res.json()
        except Exception as e:
            print(f"Failed to reset environment for task {task}: {e}")
            continue

        # Initialize the agent's memory OUTSIDE the step loop!
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Senior Python Engineer diagnosing a broken codebase. DO NOT guess the bug or hallucinate generic 'add_numbers' code. You must read the environment.\n"
                    "CRITICAL DEBUGGING WORKFLOW (FOLLOW EXACTLY):\n"
                    "1. Your VERY FIRST action MUST be to run 'pytest' to see the actual test failures and identify the broken files.\n"
                    "2. Next, use 'run_command' with 'cat <filename>' to read the full code of the failing file.\n"
                    "3. Use 'edit_file' to fix the logic. WARNING: 'edit_file' COMPLETELY OVERWRITES THE FILE. Your 'new_content' MUST include the ENTIRE script (all original imports, unchanged functions, and your fix).\n"
                    "4. Run 'pytest' again to verify all tests pass.\n"
                    "You must output ONLY valid JSON in one of the following formats:\n"
                    "{\"action_type\": \"run_command\", \"command\": \"<your command>\"}\n"
                    "OR\n"
                    "{\"action_type\": \"edit_file\", \"file_path\": \"<path>\", \"new_content\": \"<content>\"}\n"
                    "Do not include markdown formatting, explanations, or any other text."
                )
            }
        ]
        
        # 2. Step Loop
        for step in range(max_steps):
            print(f"[STEP] {step}")

            # Append the environment's response to the agent's memory
            messages.append({
                "role": "user",
                "content": f"Current Observation: {json.dumps(obs)}"
            })

            # Call LLM using our dual-routed function
           # Call LLM using our dual-routed function (WITH HACKATHON OVERRIDES)
            try:
                # FORCE the agent to read the errors and code before it can guess!
                if step == 0:
                    action_payload = {"action_type": "run_command", "command": "pytest"}
                    print("      [!] System Override: Forcing Step 0 -> pytest")
                elif step == 1:
                    action_payload = {"action_type": "run_command", "command": "cat main.py"}
                    print("      [!] System Override: Forcing Step 1 -> cat main.py")
                else:
                    # Now that it has the context, let the smart 70B model fix it
                    raw_action = get_llm_action(messages)

                    if raw_action.startswith("```json"):
                        raw_action = raw_action[7:]
                    if raw_action.startswith("```"):
                        raw_action = raw_action[3:]
                    if raw_action.endswith("```"):
                        raw_action = raw_action[:-3]
                    
                    action_payload = json.loads(raw_action.strip())
                
                print(f"      Attempting Action: {action_payload}")
                
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(action_payload)
                })
            
            except json.JSONDecodeError:
                action_payload = {"action_type": "run_command", "command": "echo 'Invalid JSON. Retrying...'"}
                print(f"      [!] JSON Parse Error. Falling back to echo.")
            except Exception as e:
                print(f"      [!] Critical Error during LLM inference: {e}")
                break

            # Execute action via HTTP POST /step
            try:
                step_res = requests.post(f"{ENV_URL}/step", json={"action": action_payload})
                step_res.raise_for_status()
                obs = step_res.json()
            except Exception as e:
                print(f"      [!] Error calling /step endpoint: {e}")
                break

            # Break loop if environment signals the task is complete
            if obs.get("done"):
                print("      [✓] Task complete signal received from environment.")
                break

        # 3. Grade the episode via HTTP
        try:
            grade_res = requests.post(f"{ENV_URL}/grade", json={"task_id": task})
            grade_res.raise_for_status()
            grade_data = grade_res.json()
            
            # Formatted exactly to spec
            print(f"[END] Task: {task} | Reward: {grade_data.get('total_reward')} | Success: {grade_data.get('success')} | Breakdown: {grade_data.get('breakdown')}")
        except Exception as e:
            print(f"Error calling /grade endpoint for task {task}: {e}")

if __name__ == "__main__":
    try:
        print("Starting inference protocol...")
        run_inference()
    except Exception as e:
        print(f"CRITICAL INFERENCE CRASH: {e}")
        import traceback
        traceback.print_exc()
        # Exit cleanly so the evaluator captures the logs
        import sys
        sys.exit(1)
