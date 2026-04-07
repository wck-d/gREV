from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    current_directory: str = Field(
        ...,
        description="Absolute path of the agent's current working directory.",
    )
    directory_contents: list[str] = Field(
        ...,
        description="List of files and folders in the current directory.",
    )
    last_command_stdout: str = Field(
        ...,
        description="Captured standard output from the most recent shell command.",
    )
    last_command_stderr: str = Field(
        ...,
        description="Captured standard error from the most recent shell command.",
    )
    # --- Added to satisfy OpenEnv alpha serialization quirks ---
    reward: float = Field(
        default=0.0,
        description="Current reward value for the state.",
    )
    done: bool = Field(
        default=False,
        description="Flag indicating if the task is complete.",
    )


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action_type: Literal["run_command", "edit_file"] = Field(
        ...,
        description="Type of action the agent wants to execute.",
    )
    command: str | None = Field(
        default=None,
        description="Shell command to execute when action_type is 'run_command'.",
    )
    file_path: str | None = Field(
        default=None,
        description="Path of the file to edit when action_type is 'edit_file'.",
    )
    new_content: str | None = Field(
        default=None,
        description="Replacement content to write when action_type is 'edit_file'.",
    )


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fractional progress score from 0.0 to 1.0.",
    )
