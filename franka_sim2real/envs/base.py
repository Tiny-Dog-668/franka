from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import RobotAction, RobotObservation


class BaseFrankaEnv(ABC):
    backend_name: str

    @abstractmethod
    def reset(self, episode_index: int = 0) -> RobotObservation:
        ...

    @abstractmethod
    def step(self, action: RobotAction) -> tuple[RobotObservation, float, bool, dict[str, Any]]:
        ...

    def close(self) -> None:
        return None
