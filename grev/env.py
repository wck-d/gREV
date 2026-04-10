"""gREV environment implementation — RepoRescueEnv.

A deterministic RL sandbox where agents diagnose and fix broken Python repos.
Grading uses a multi-component weighted reward inspired by real code-review quality metrics.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    try:
        from openenv_core.env_server.interfaces import Environment
    except ImportError:
        # Fallback for local dev without openenv installed
        Environment = object

try:
    from grev.models import GrevAction, GrevObservation, GrevState
except ImportError:
    from models import GrevAction, GrevObservation, GrevState


WORKSPACE_DIR = "/tmp/grev_workspace"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


# ── Task configurations ─────────────────────────────────────

@dataclass(frozen=True)
class TaskConfig:
    task_level: str
    max_steps: int
    bug_count: int        # number of distinct bugs in the task
    test_count: int       # total tests in the suite
    time_budget: float    # abstract time units the agent has


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_level="easy",
        max_steps=12,
        bug_count=2,
        test_count=8,
        time_budget=12.0,
    ),
    "medium": TaskConfig(
        task_level="medium",
        max_steps=16,
        bug_count=3,
        test_count=14,
        time_budget=16.0,
    ),
    "hard": TaskConfig(
        task_level="hard",
        max_steps=20,
        bug_count=4,
        test_count=15,
        time_budget=20.0,
    ),
    "medium_hard": TaskConfig(
        task_level="medium_hard",
        max_steps=18,
        bug_count=3,
        test_count=16,
        time_budget=18.0,
    ),
    "very_hard": TaskConfig(
        task_level="very_hard",
        max_steps=24,
        bug_count=4,
        test_count=27,
        time_budget=24.0,
    ),
}

# Reward component weights — rebalanced for cross-model consistency
# test_pass_rate gets majority weight as the most deterministic signal
REWARD_WEIGHTS = {
    "test_pass_rate": 0.60,      # primary signal — fully deterministic
    "diagnosis_quality": 0.20,   # did the agent investigate before editing?
    "fix_efficiency": 0.10,      # gentle tiebreaker, never zeros out a correct fix
    "penalty_avoidance": 0.10,   # clean execution bonus
}


# ── Grader ───────────────────────────────────────────────────

class RepairGrader:
    """Multi-component grader for code repair quality."""

    def __init__(self, config: TaskConfig):
        self.config = config
        self._penalties: float = 0.0
        self._files_read: set[str] = set()
        self._files_edited: set[str] = set()
        self._pytest_runs: int = 0
        self._invalid_actions: int = 0
        self._timeout_count: int = 0
        self._best_pass_rate: float = 0.0
        # Stall tracking: consecutive steps with unchanged pass rate and no edit
        self._stall_streak: int = 0
        self._last_recorded_rate: float = -1.0  # sentinel
        self._stall_penalties: float = 0.0

    def record_read(self, filename: str) -> None:
        self._files_read.add(filename)

    def record_edit(self, filename: str) -> None:
        self._files_edited.add(filename)
        # Any edit resets the stall streak
        self._stall_streak = 0

    def record_pytest_run(self) -> None:
        self._pytest_runs += 1

    def record_stall(self, current_pass_rate: float) -> None:
        """Call once per step to detect progress-free loops.

        If pass rate hasn't changed and no edit was made in this step,
        increment the stall streak. After 3 consecutive stall steps,
        apply a small penalty per additional stale step (0.04 per step).
        Resets when pass rate improves or an edit is recorded.
        """
        if current_pass_rate > self._last_recorded_rate:
            # Progress made — reset stall streak
            self._stall_streak = 0
        else:
            self._stall_streak += 1
            # Grace period: 3 stall steps before any penalty
            if self._stall_streak > 3:
                penalty = 0.04
                self._stall_penalties += penalty
                self._penalties += penalty

        self._last_recorded_rate = current_pass_rate

    def record_invalid_action(self) -> None:
        self._invalid_actions += 1
        self._penalties += 0.05

    def record_timeout(self) -> None:
        self._timeout_count += 1
        self._penalties += 0.10

    def update_best_pass_rate(self, rate: float) -> None:
        self._best_pass_rate = max(self._best_pass_rate, rate)

    def test_pass_rate_score(self, passed: int, total: int) -> float:
        """Primary grading component: fraction of tests passing.

        Uses config.test_count as fallback denominator when pytest can't collect
        (e.g. SyntaxError prevents collection). This ensures consistent scoring
        regardless of how the model approaches the task.
        """
        # Fallback: if pytest reported 0 total (collection failure), use known count
        effective_total = total if total > 0 else self.config.test_count
        if effective_total == 0:
            return 0.0
        return _clamp(passed / effective_total)

    def diagnosis_quality_score(self) -> float:
        """Did the agent investigate properly before editing?

        Smooth 0-1 scale rewarding breadth of investigation.
        Model-agnostic: doesn't reward any specific action sequence.
        """
        score = 0.0
        if self._pytest_runs >= 1:        # ran pytest to see what's broken
            score += 0.35
        if len(self._files_read) >= 1:    # read at least one source file
            score += 0.30
        if len(self._files_read) >= 2:    # cross-file investigation
            score += 0.20
        if len(self._files_edited) >= 1:  # attempted a fix
            score += 0.15
        return _clamp(score)

    def fix_efficiency_score(self, steps_taken: int) -> float:
        """Efficiency tiebreaker — never punishes slow-but-correct models.

        Only applied when agent achieved >=50% pass rate (made real progress).
        Decays from 1.0 to 0.5 across the full budget, so a model using every
        step still gets 0.5 here — efficiency can only help, not destroy.
        """
        if self._best_pass_rate < 0.5:
            return 0.0  # no efficiency credit without meaningful progress
        max_steps = self.config.max_steps
        # Floors at 0.5: even at max_steps, efficiency = 1.0 - 0.5*(1.0) = 0.5
        efficiency = 1.0 - 0.5 * (steps_taken / max_steps)
        return _clamp(efficiency)

    def penalty_avoidance_score(self) -> float:
        """Higher score for clean execution without errors."""
        return _clamp(1.0 - self._penalties)

    def aggregate_score(self, passed: int, total: int, steps_taken: int) -> Dict[str, float]:
        """Compute weighted composite score with component breakdown."""
        components = {
            "test_pass_rate": self.test_pass_rate_score(passed, total),
            "diagnosis_quality": self.diagnosis_quality_score(),
            "fix_efficiency": self.fix_efficiency_score(steps_taken),
            "penalty_avoidance": self.penalty_avoidance_score(),
        }

        total_score = sum(
            REWARD_WEIGHTS[k] * components[k] for k in REWARD_WEIGHTS
        )
        components["total"] = _clamp(total_score)
        return components


# ── Environment ──────────────────────────────────────────────

class gREVEnv(Environment):
    """OpenEnv-compliant environment for broken repository repair."""

    def __init__(self, **kwargs):
        self._task_level: str = "easy"
        self._config: TaskConfig = TASK_CONFIGS["easy"]
        self._step_count: int = 0
        self._done: bool = False
        self._grader: RepairGrader = RepairGrader(self._config)
        self._last_passed: int = 0
        self._last_total: int = 0
        self._prev_pass_rate: float = 0.0

    @property
    def state(self) -> GrevState:
        return GrevState(
            task_level=self._task_level,
            step_count=self._step_count,
            workspace_dir=WORKSPACE_DIR,
            max_steps=self._config.max_steps,
            directory_contents=self._get_dir_contents(),
        )

    def reset(self, task_level: str = "easy", seed: int = 42, **kwargs) -> GrevObservation:
        """Wipe workspace and copy fresh task files."""
        self._task_level = task_level
        self._config = TASK_CONFIGS.get(task_level, TASK_CONFIGS["easy"])
        self._step_count = 0
        self._done = False
        self._grader = RepairGrader(self._config)
        self._last_passed = 0
        self._last_total = 0
        self._prev_pass_rate = 0.0

        # Wipe workspace
        if os.path.exists(WORKSPACE_DIR):
            shutil.rmtree(WORKSPACE_DIR)

        # Locate and copy task source
        task_source = self._find_task_source(task_level)
        if task_source:
            shutil.copytree(task_source, WORKSPACE_DIR)
        else:
            os.makedirs(WORKSPACE_DIR, exist_ok=True)

        return GrevObservation(
            done=False,
            reward=0.0,
            current_directory=WORKSPACE_DIR,
            directory_contents=self._get_dir_contents(),
            last_command_stdout=f"Environment reset to {task_level}. "
                                f"Workspace contains a broken Python project. "
                                f"Your goal: fix the code so all pytest tests pass.",
            last_command_stderr="",
            last_error=None,
        )

    def step(self, action: GrevAction) -> GrevObservation:
        """Execute the agent's action and return intermediate reward."""
        stdout = ""
        stderr = ""
        error: Optional[str] = None
        step_reward = 0.0

        self._step_count += 1

        try:
            if action.action_type == "run_command":
                if not action.command:
                    error = "Missing command for run_command action."
                    stderr = error
                    self._grader.record_invalid_action()
                else:
                    stdout, stderr, error = self._execute_command(action.command)
            elif action.action_type == "edit_file":
                stdout, stderr, error = self._execute_edit(action)
            else:
                error = f"Unknown action_type: {action.action_type}"
                stderr = error
                self._grader.record_invalid_action()
        except subprocess.TimeoutExpired:
            error = "Command timed out after 15 seconds."
            stderr = error
            self._grader.record_timeout()
        except Exception as e:
            error = str(e)
            stderr = error

        # ── Compute intermediate reward ──────────────────────
        passed, total = self._run_pytest_silent()
        self._last_passed = passed
        self._last_total = total

        # Use config.test_count as denominator fallback for collection failures
        effective_total = total if total > 0 else self._config.test_count
        current_pass_rate = passed / effective_total if effective_total > 0 else 0.0
        self._grader.update_best_pass_rate(current_pass_rate)
        self._grader.record_stall(current_pass_rate)  # stall penalty if looping

        # Composite reward — deterministic, no path-dependent delta shaping
        components = self._grader.aggregate_score(passed, total, self._step_count)
        step_reward = _clamp(components["total"])

        # Check done conditions
        done = False
        if current_pass_rate == 1.0:
            done = True  # all tests pass
        if self._step_count >= self._config.max_steps:
            done = True  # budget exhausted

        self._done = done

        return GrevObservation(
            done=done,
            reward=step_reward,
            current_directory=WORKSPACE_DIR,
            directory_contents=self._get_dir_contents(),
            last_command_stdout=stdout,
            last_command_stderr=stderr,
            last_error=error,
        )

    def grade(self) -> Tuple[float, Dict]:
        """Final grading: run pytest and compute component scores."""
        passed, total = self._run_pytest_silent()
        self._last_passed = passed
        self._last_total = total

        components = self._grader.aggregate_score(passed, total, self._step_count)

        return components["total"], {
            "passed": passed,
            "failed": total - passed,
            "total": total,
            "components": components,
        }

    def close(self):
        pass

    # ── Action handlers ──────────────────────────────────────

    def _execute_command(self, command: str) -> Tuple[str, str, Optional[str]]:
        """Run a shell command in the workspace sandbox."""
        # Track diagnostic actions
        if "cat " in command or "head " in command or "less " in command:
            # Extract filename from command
            parts = command.split()
            if len(parts) >= 2:
                self._grader.record_read(parts[-1])

        if "pytest" in command:
            self._grader.record_pytest_run()

        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout or "", result.stderr or "", None

    def _execute_edit(self, action: GrevAction) -> Tuple[str, str, Optional[str]]:
        """Write new content to a file."""
        if not action.file_path:
            self._grader.record_invalid_action()
            return "", "Missing file_path for edit_file action.", "Missing file_path"
        if action.new_content is None:
            self._grader.record_invalid_action()
            return "", "Missing new_content for edit_file action.", "Missing new_content"

        target_path = self._resolve_workspace_path(action.file_path)
        self._grader.record_edit(action.file_path)

        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(action.new_content)
            return f"File {action.file_path} updated successfully.", "", None
        except Exception as e:
            self._grader.record_invalid_action()
            return "", str(e), str(e)

    # ── Helpers ──────────────────────────────────────────────

    def _find_task_source(self, task_level: str) -> Optional[str]:
        candidates = [
            f"tasks/{task_level}",
            f"/app/env/tasks/{task_level}",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "tasks", task_level),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def _get_dir_contents(self) -> List[str]:
        if not os.path.exists(WORKSPACE_DIR):
            return []
        return os.listdir(WORKSPACE_DIR)

    def _resolve_workspace_path(self, file_path: str) -> str:
        clean_path = os.path.normpath(file_path).lstrip("/")
        if clean_path.startswith("tmp/grev_workspace"):
            clean_path = clean_path.replace("tmp/grev_workspace", "", 1).lstrip("/")
        return os.path.join(WORKSPACE_DIR, clean_path)

    def _run_pytest_silent(self) -> Tuple[int, int]:
        """Run pytest silently and return (passed, total)."""
        try:
            result = subprocess.run(
                f"{sys.executable} -m pytest -v --tb=no -q",
                shell=True,
                cwd=WORKSPACE_DIR,
                capture_output=True,
                text=True,
                timeout=10,
            )
            stdout = result.stdout or ""
            return self._parse_pytest_counts(stdout)
        except Exception:
            return 0, 0

    @staticmethod
    def _parse_pytest_counts(stdout: str) -> Tuple[int, int]:
        """Parse 'X passed, Y failed' from pytest output."""
        passed = 0
        failed = 0
        error_count = 0

        passed_match = re.search(r"(\d+)\s+passed", stdout)
        failed_match = re.search(r"(\d+)\s+failed", stdout)
        error_match = re.search(r"(\d+)\s+error", stdout)

        if passed_match:
            passed = int(passed_match.group(1))
        if failed_match:
            failed = int(failed_match.group(1))
        if error_match:
            error_count = int(error_match.group(1))

        total = passed + failed + error_count
        return passed, total
