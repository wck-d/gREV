"""Typed models for gREV (RepoRescueEnv)."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    try:
        from openenv_core.env_server.types import Action, Observation, State
    except ImportError:
        # Fallback for local dev without openenv installed
        Action = BaseModel
        Observation = BaseModel
        State = BaseModel


class GrevAction(Action):
    """Action submitted by the coding agent."""

    action_type: Literal["run_command", "edit_file"] = Field(
        ...,
        description="Type of action the agent wants to execute.",
    )
    command: Optional[str] = Field(
        default=None,
        description="Shell command to execute when action_type is 'run_command'.",
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Path of the file to edit when action_type is 'edit_file'.",
    )
    new_content: Optional[str] = Field(
        default=None,
        description="Replacement content to write when action_type is 'edit_file'.",
    )


class GrevObservation(Observation):
    """Observation returned after each step."""

    # These fields are provided by the OpenEnv Observation base class,
    # but we declare them explicitly so the model works both with and
    # without the openenv package installed.
    done: bool = Field(default=False, description="Whether the episode is complete.")
    reward: float = Field(default=0.0, description="Reward for this step.")

    current_directory: str = Field(
        default="",
        description="Absolute path of the agent's current working directory.",
    )
    directory_contents: List[str] = Field(
        default_factory=list,
        description="List of files and folders in the current directory.",
    )
    last_command_stdout: str = Field(
        default="",
        description="Captured standard output from the most recent shell command.",
    )
    last_command_stderr: str = Field(
        default="",
        description="Captured standard error from the most recent shell command.",
    )
    last_error: Optional[str] = Field(
        default=None,
        description="Error message from the last action, if any.",
    )


class GrevState(State):
    """Full environment state exposed through state()."""

    task_level: str = ""
    step_count: int = 0
    workspace_dir: str = ""
    max_steps: int = 0
    directory_contents: List[str] = Field(default_factory=list)
