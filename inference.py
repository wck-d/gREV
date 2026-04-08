import os
import json
import requests
import re
import sys
from openai import OpenAI

# --- Configuration & Environment Variables ---
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# Quick sanitize to strip Markdown if it accidentally gets passed into the terminal export
if ENV_URL.startswith("[") and "](" in ENV_URL:
    match = re.search(r'\((https?://.*?)\)', ENV_URL)
    if match:
        ENV_URL = match.group(1)

# API Keys
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

hf_client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)
HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"

def get_llm_action(messages):
    """Forces Groq's 70B model first for maximum intelligence."""
    if GROQ_API_KEY:
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1 
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"      [!] Groq API failed: {e}. Trying Hugging Face...")
    
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
    # Map tasks to their primary files so we don't 'cat' the wrong file
    task_to_file = {
        "easy": "main.py",
        "medium": "parser.py",
        "hard": "fetcher.py"
    }
    max_steps = 10

    for task in tasks:
        print(f"\n[START] Task: {task}")
        target_file = task_to_file.get(task, "main.py")

        try:
            reset_res = requests.post(f"{ENV_URL}/reset", json={"task_id": task})
            reset_res.raise_for_status()
            obs = reset_res.json()
        except Exception as e:
            print(f"Failed to reset environment for task {task}: {e}")
            continue

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Senior Python Engineer diagnosing a broken codebase. DO NOT guess the bug.\n"
                    "1. Run 'pytest' to see failures.\n"
                    "2. Use 'run_command' with 'cat <filename>' to read the code.\n"
                    "3. Use 'edit_file' to fix logic. 'edit_file' OVERWRITES the file; include ALL code.\n"
                    "Output ONLY valid JSON:\n"
                    "{\"action_type\": \"run_command\", \"command\": \"<cmd>\"}\n"
                    "{\"action_type\": \"edit_file\", \"file_path\": \"<path>\", \"new_content\": \"<content>\"}"
                )
            }
        ]
        
        for step in range(max_steps):
            print(f"[STEP] {step}")

            messages.append({
                "role": "user",
                "content": f"Current Observation: {json.dumps(obs)}"
            })

            try:
                # FIX: Use target_file variable so we cat the correct file for the current task
                if step == 0:
                    action_payload = {"action_type": "run_command", "command": "pytest"}
                    print("      [!] System Override: Forcing Step 0 -> pytest")
                elif step == 1:
                    action_payload = {"action_type": "run_command", "command": f"cat {target_file}"}
                    print(f"      [!] System Override: Forcing Step 1 -> cat {target_file}")
                else:
                    raw_action = get_llm_action(messages)
                    
                    # Robust JSON cleaning (regex handles nested backticks or messy prefixes)
                    json_match = re.search(r'\{.*\}', raw_action, re.DOTALL)
                    if json_match:
                        raw_action = json_match.group(0)
                    
                    action_payload = json.loads(raw_action.strip())
                
                print(f"      Attempting Action: {action_payload}")
                
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(action_payload)
                })
            
            except (json.JSONDecodeError, AttributeError):
                action_payload = {"action_type": "run_command", "command": "ls -R"}
                print(f"      [!] JSON Parse Error. Falling back to discovery.")
            except Exception as e:
                print(f"      [!] Critical Error: {e}")
                break

            try:
                step_res = requests.post(f"{ENV_URL}/step", json={"action": action_payload})
                step_res.raise_for_status()
                obs = step_res.json()
            except Exception as e:
                print(f"      [!] Error calling /step: {e}")
                break

            if obs.get("done"):
                print("      [✓] Task complete signal received.")
                break

        # Final Grading
        try:
            grade_res = requests.post(f"{ENV_URL}/grade", json={"task_id": task})
            grade_res.raise_for_status()
            grade_data = grade_res.json()
            print(f"[END] Task: {task} | Reward: {grade_data.get('total_reward')} | Success: {grade_data.get('success')}")
        except Exception as e:
            print(f"Error calling /grade: {e}")

if __name__ == "__main__":
    try:
        run_inference()
    except Exception as e:
        print(f"CRITICAL INFERENCE CRASH: {e}")
        sys.exit(1)
