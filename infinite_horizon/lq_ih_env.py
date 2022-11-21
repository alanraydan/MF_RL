import torch
import numpy as np

class LqIhEnv:

    def __init__(self, c1, c2, c3, c4, c5, beta, sigma, mu0, dt):
        # MF environment parameters
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5
        self.beta = beta
        self.discount = np.exp(-beta * dt)
        self.sigma = sigma
        self.mu0 = mu0

        # Discretization parameters
        self.dt = dt
        self.t = None
        self.time_step = None
    
    def reset(self):
        self.t = 0
        self.time_step = 0
        x = self.mu0.sample()
        return x

    def step(self, x, alpha, m):

        cost = (
                0.5 * alpha**2
                + self.c1 * (x - self.c2 * m)**2
                + self.c3 * (x - self.c4)**2
                + self.c5 * m**2
            ) * self.dt

        dW = np.random.normal(loc=0.0, scale=np.sqrt(self.dt))
        x_new = x + alpha * self.dt + self.sigma * dW

        self.t += self.dt
        self.time_step += 1

        return x_new, cost

    # -------------------------------------------------------------------------
    def optimal_control_mfg(self, x):
        gamma2 = (-self.beta + torch.sqrt(self.beta**2 + 8 * (self.c1 + self.c3))) / 4
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
        numerator = 2 * self.c5 * m \
            - 2 * self.c1 * self.c2 * m * (2 - self.c2) \
            - 2 * self.c3 * self.c4
        denominator = self.beta + 2 * gamma2
        gamma1 = numerator / denominator
        optimal_control = -(2 * gamma2 * x + gamma1)
        return optimal_control

    def optimal_mean_mfc(self):
        gamma2 = (-self.beta + torch.sqrt(self.beta ** 2 + 8 * (self.c1 + self.c3))) / 4
        numerator = self.c3 * self.c4
        denominator = gamma2 * (self.beta + 2 * gamma2) \
            + self.c5 \
            - self.c1 * self.c2 * (2 - self.c2)
        return numerator/denominator
