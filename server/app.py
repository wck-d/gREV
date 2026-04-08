import subprocess
import re
import sys
from fastapi import Request
from openenv.core.env_server import create_app
from grev.env import gREVEnv
from grev.models import Action, Observation  # <--- Add this import!

# 1. Initialize the OpenEnv app with the required Pydantic models
app = create_app(gREVEnv, Action, Observation) # <--- Pass them here!


@app.get("/")
async def root_health():
    return JSONResponse(content={"status": "healthy", "message": "gREV is alive!"})

@app.get("/health")
async def explicit_health():
    return JSONResponse(content={"status": "healthy"})

# ==========================================
# 2. HACKATHON SURVIVAL: ROUTE HIJACKING
# Strip the framework's broken default /grade route so it stops ignoring our workspace
# ==========================================
for route in list(app.routes):
    if hasattr(route, "path") and route.path == "/grade":
        app.routes.remove(route)

# 3. Inject our Deterministic, Workspace-Aware Grader
@app.post("/grade")
async def hackathon_grader(request: Request):
    try:
        # We explicitly target the workspace where the AI just wrote the code!
        result = subprocess.run(
            f"{sys.executable} -m pytest", 
            shell=True, 
            cwd="/tmp/grev_workspace", 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        stdout = result.stdout or ""
        
        # Perfect Score
        if result.returncode == 0:
            return {
                "total_reward": 1.0, 
                "success": True, 
                "steps_taken": 10,
                "breakdown": {"status": "all_tests_passed"}
            }
            
        # Partial Score Parsing (Meaningful Reward Shaping)
        passed = 0
        failed = 0
        passed_match = re.search(r"(\d+)\s+passed", stdout)
        failed_match = re.search(r"(\d+)\s+failed", stdout)
        
        if passed_match: 
            passed = int(passed_match.group(1))
        if failed_match: 
            failed = int(failed_match.group(1))
            
        total = passed + failed
        score = passed / total if total > 0 else 0.0
        
        return {
            "total_reward": float(score),
            "success": False,
            "steps_taken": 10,
            "breakdown": {"passed": passed, "failed": failed}
        }
    except Exception as e:
        return {
            "total_reward": 0.0, 
            "success": False, 
            "breakdown": {"error": str(e)}
        }
