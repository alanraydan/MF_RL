from abc import ABC, abstractmethod
from running_cost_func import RunningCostFunc, LqRunningCostFunc
from sde import ControlledSde
import torch
import numpy as np


class MfEnv(ABC):
    """
    Mean Field Environment base class
    """
    @abstractmethod
    def __init__(
        self,
        initial_distribution: torch.distributions,
        beta: float,
        running_cost: RunningCostFunc,
        sde: ControlledSde,
        dt: float
    ) -> None:
        self.initial_distribution = initial_distribution
        self.beta = beta
        self.discount = np.exp(-beta * dt)
        self.running_cost = running_cost
        self.sde = sde
        self.dt = dt

    @abstractmethod
    def reset(self) -> torch.tensor:
        pass

    @abstractmethod
    def step(self, state: torch.tensor, action: torch.tensor, mean_field: torch.distributions) -> torch.tensor:
        pass


class IhMfEnv(MfEnv):
    """
    Infinite Horizon Mean Field Environment
    """
    def __init__(
        self,
        initial_distribution: torch.distributions,
        beta: float,
        running_cost: LqRunningCostFunc,
        sde: ControlledSde,
        dt: float,
    ) -> None:
        super(IhMfEnv, self).__init__(initial_distribution, beta, running_cost, sde, dt)

    def reset(self) -> torch.tensor:
        return self.initial_distribution.sample()

    def step(self, state: torch.tensor, action: torch.tensor, mean_field: torch.distributions):
        cost = self.running_cost(state, action, mean_field) * self.dt
        next_state = self.sde.evolve(self.dt, state, action, mean_field)
        return next_state, cost

    # --Benchmark solutions--
    def optimal_control_mfg(self, x):
        gamma2 = (-self.beta + np.sqrt(self.beta**2 + 8*(self.running_cost.c1 + self.running_cost.c3)))/4
        m = self.optimal_mean_mfg()
        numerator = -(2 * self.running_cost.c1 * self.running_cost.c2 * m
                      + 2 * self.running_cost.c3 * self.running_cost.c4)
        denominator = self.beta + 2 * gamma2
        gamma1 = numerator / denominator
        optimal_control = -(2 * gamma2 * x + gamma1)
        return optimal_control

    def optimal_mean_mfg(self):
        gamma2 = (-self.beta + np.sqrt(self.beta ** 2 + 8 * (self.running_cost.c1 + self.running_cost.c3))) / 4
        numerator = self.running_cost.c3 * self.running_cost.c4
        denominator = gamma2 * (self.beta + 2 * gamma2) - self.running_cost.c1 * self.running_cost.c2
        return numerator / denominator

    def optimal_control_mfc(self, x):
        gamma2 = (-self.beta + np.sqrt(self.beta**2 + 8*(self.running_cost.c1 + self.running_cost.c3)))/4
        m = self.optimal_mean_mfc()
        numerator = 2 * self.running_cost.c5 * m \
                    - 2 * self.running_cost.c1 * self.running_cost.c2 * m * (2 - self.running_cost.c2) \
                    - 2 * self.running_cost.c3 * self.running_cost.c4
        denominator = self.beta + 2 * gamma2
        gamma1 = numerator / denominator
        optimal_control = -(2 * gamma2 * x + gamma1)
        return optimal_control

    def optimal_mean_mfc(self):
        gamma2 = (-self.beta + np.sqrt(self.beta ** 2 + 8 * (self.running_cost.c1 + self.running_cost.c3))) / 4
        numerator = self.running_cost.c3 * self.running_cost.c4
        denominator = gamma2 * (self.beta + 2 * gamma2) + self.running_cost.c5 - self.running_cost.c1 * self.running_cost.c2 * (2 - self.running_cost.c2)
        return numerator/denominator




