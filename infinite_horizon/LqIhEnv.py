"""
MDP discretization of MEAN FIELD LINEAR QUADRATIC environment
"""
import torch


class LqIhEnv:
    """
    Description:
        This class models the state dynamics and running reward for a
        LINEAR-QUADRATIC INFINITE HORIZON (LqIh) stochastic control environment
        similarly to OpenAI's gym API. The problem and the
        parameters are taken from the thesis of Andrea Angiuli.
    """

    def __init__(self):
        self.time_step = None
        self.c1 = torch.tensor([[0.25]])
        self.c2 = torch.tensor([[1.5]])
        self.c3 = torch.tensor([[0.5]])
        self.c4 = torch.tensor([[0.6]])
        self.c5 = torch.tensor([[1.0]])
        self.beta = torch.tensor([[1.0]])
        self.sigma = torch.tensor([[0.3]])
        self.T = 20  # Infinity horizon truncated at T >> 0
        self.dt = torch.tensor([[1e-2]])

        self.done = None
        self.x = None

    def reset(self, x_init):
        self.time_step = 0
        self.done = False
        self.x = x_init
        return self.x

    def step(self, action, state_mean):
        with torch.no_grad():
            cost = self.compute_cost(action, state_mean) * self.dt
            reward = -cost
            x_new, noise = self.euler_maruyama(action)

        self.time_step += 1
        self.x = x_new
        return x_new, reward, noise

    # ----------------------------------------------------------------------

    # --Helper functions--
    def euler_maruyama(self, action):
        dw = torch.normal(mean=0.0, std=torch.sqrt(self.dt))
        x_new = self.x + action * self.dt + self.sigma * dw
        return x_new, dw

    def compute_cost(self, action, mean):
        return 0.5 * action ** 2 \
               + self.c1 * (self.x - self.c2 * mean) ** 2 \
               + self.c3 * (self.x - self.c4) ** 2 \
               + self.c5 * mean ** 2

    # --Benchmark solutions--
    def optimal_control_mfg(self, x):
        gamma2 = (-self.beta + torch.sqrt(self.beta**2 + 8*(self.c1 + self.c3)))/4
        m = self.optimal_mean_mfg()
        numerator = -(2 * self.c1 * self.c2 * m + 2 * self.c3 * self.c4)
        denominator = self.beta + 2 * gamma2
        gamma1 = numerator / denominator
        optimal_control = -(2 * gamma2 * x + gamma1)
        return optimal_control

    def optimal_mean_mfg(self):
        gamma2 = (-self.beta + torch.sqrt(self.beta ** 2 + 8 * (self.c1 + self.c3))) / 4
        numerator = self.c3 * self.c4
        denominator = gamma2 * (self.beta + 2 * gamma2) - self.c1 * self.c2
        return numerator / denominator

    def optimal_control_mfc(self, x):
        gamma2 = (-self.beta + torch.sqrt(self.beta**2 + 8*(self.c1 + self.c3)))/4
        m = self.optimal_mean_mfc()
        numerator = 2 * self.c5 * m - 2 * self.c1 * self.c2 * m * (2 - self.c2) - 2 * self.c3 * self.c4
        denominator = self.beta + 2 * gamma2
        gamma1 = numerator / denominator
        optimal_control = -(2 * gamma2 * x + gamma1)
        return optimal_control

    def optimal_mean_mfc(self):
        gamma2 = (-self.beta + torch.sqrt(self.beta ** 2 + 8 * (self.c1 + self.c3))) / 4
        numerator = self.c3 * self.c4
        denominator = gamma2 * (self.beta + 2 * gamma2) + self.c5 - self.c1 * self.c2 * (2 - self.c2)
        return numerator/denominator
