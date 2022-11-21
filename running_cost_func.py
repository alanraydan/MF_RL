from abc import ABC, abstractmethod
import torch


class RunningCostFunc(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, state: torch.tensor, action: torch.tensor, mean_field: torch.distributions) -> torch.tensor:
        pass


class LqRunningCostFunc(RunningCostFunc):
    """
    Linear Quadratic Running Cost Function
    """
    def __init__(
        self,
        c1: float,
        c2: float,
        c3: float,
        c4: float,
        c5: float
    ) -> None:
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5

    def __call__(self, state: torch.tensor, action: torch.tensor, mean_field: torch.distributions) -> torch.tensor:
        return 0.5 * action**2 \
            + self.c1 * (state - self.c2 * mean_field.mean)**2 \
            + self.c3 * (state - self.c4)**2 \
            + self.c5 * mean_field.mean**2
