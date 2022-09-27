"""
MDP discretization of LQ environment satisfying OpenAI Gym's Env API
Be sure to install gym and scipy
"""

import gym
from gym.utils import seeding
import numpy as np
from scipy import integrate

# If you have stable_baselines3 installed you can
# use the following module to check if your environment
# uses the correct API:
# from stable_baselines3.common.env_checker import check_env


class PriceImpactEnv(gym.Env):
    """
    Description:
        This class models the evolution of a linear quadratic
        Mean Field Game problem where mean field interaction is through
        the first moment of the control process. The specific problem is the Price Impact
        model from Carmona & Delarue (2018)
    """
    def __init__(self, control_distribution):
        # parameters
        self.control_distribution = control_distribution
        self.np_random = None
        self.time_step = 0
        self.total_steps = 100
        self.t_init = 0.0
        self.t_final = 1.0
        self.dt = (self.t_final - self.t_init) / self.total_steps
        self.x0_mean = 0.5
        self.x0_std = 0.3
        self.c_x = 2.0
        self.c_g = 0.3
        self.c_alpha = 1.0
        self.gamma = 1.75
        self.sigma = 0.5
        self.x_min = -1.5
        self.x_max = 1.75
        self.action_min = -4.0
        self.action_max = 6.0

        # state variables
        self.t = None
        self.x = None
        self.done = None

        # setup action and state spaces
        lows = np.array([self.t_init, self.x_min])
        highs = np.array([self.t_final, self.x_max])
        self.observation_space = gym.spaces.Box(low=lows, high=highs)
        self.action_space = gym.spaces.Box(low=self.action_min, high=self.action_max, shape=(1,))
        self.seed()

    def reset(self):
        self.time_step = 0
        self.done = False
        self.t = self.t_init
        self.x = self.np_random.normal(loc=self.x0_mean, scale=self.x0_std)

        state = np.array([self.t, self.x])
        return state

    def step(self, action):
        if self.t < self.t_final:
            theta = self.control_distribution[self.time_step]
            cost = self.compute_running_cost(action, theta)*self.dt
            done = False
        else:
            cost = self.compute_terminal_cost()
            done = True
        reward = -cost

        x_new = self.euler_maruyama(action)
        t_new = self.t + self.dt
        state = np.array([t_new, float(x_new)], dtype=np.float32)

        self.time_step += 1
        self.done = done
        self.t = t_new
        self.x = x_new
        return state, float(reward), done, {}

    def seed(self, seed=None):
        # I copied this from OpenAI's github or something
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ---------------------------------------------------------------------------
    # Helper functions

    def euler_maruyama(self, action):
        dw = self.np_random.normal(loc=0.0, scale=np.sqrt(self.dt))
        x_new = self.x + action*self.dt + self.sigma*dw
        return x_new

    def compute_running_cost(self, action, theta):
        return (self.c_alpha/2.0) * action ** 2 \
               + (self.c_x/2.0) * self.x ** 2 \
               - self.gamma * self.x * theta

    def compute_terminal_cost(self):
        return (self.c_g/2.0)*self.x**2

    # ---------------------------------------------------------------------------
    # MFG Helpers

    def compute_control_mean(self, t):
        eta = self.eta_bar(t)
        eta_int = integrate.quad(self.eta_bar, 0.0, t)[0]
        mu = -(self.x0_mean / self.c_alpha) * eta * np.exp(-eta_int / self.c_alpha)
        return mu

    def eta_bar(self, t):
        A = -self.gamma / (2 * self.c_alpha)
        B = 1 / self.c_alpha
        C = self.c_x
        R = A ** 2 + B * C
        delta_plus = -A + np.sqrt(R)
        delta_minus = -A - np.sqrt(R)
        exp = np.exp((delta_plus - delta_minus) * (self.t_final - t))
        numerator = -C * (exp - 1) - self.c_g * (delta_plus * exp - delta_minus)
        denominator = delta_minus * exp - delta_plus - self.c_g * B * (exp - 1)
        return numerator / denominator

    def eta(self, t):
        c_term = self.c_alpha * np.sqrt(self.c_x / self.c_alpha)
        exp_term = np.exp(2 * (self.t_final - t) * np.sqrt(self.c_x / self.c_alpha))
        numerator = c_term - self.c_g - (c_term + self.c_g) * exp_term
        denominator = c_term - self.c_g + (c_term + self.c_g) * exp_term
        return -c_term * (numerator / denominator)

    def x_bar_MFG(self, t):
        eta_int = integrate.quad(self.eta_bar, 0.0, t)[0]
        return self.x0_mean * np.exp(-eta_int / self.c_alpha)

    def optimal_MFG_control(self, t, x):
        return -(self.eta(t) * x + (self.eta_bar(t) - self.eta(t)) * self.x_bar_MFG(t)) / self.c_alpha

    # ---------------------------------------------------------------------------
    # MFC Helpers

    def phi_bar(self, t):
        R = 1 / self.c_alpha
        a = 2 * self.gamma * R
        b = R * (self.gamma**2 * R - self.c_x)
        c_1, c_2 = np.roots([1, a, b])
        exp = np.exp((self.t_final - t) * (c_2 - c_1))
        numerator = (c_2 + R*self.c_g)*c_1*exp - c_2*(c_1 + R*self.c_g)
        denominator = (c_2 + R*self.c_g)*exp - (c_1 + R*self.c_g)
        return -numerator/(R*denominator)

    def x_bar_MFC(self, t):
        phi_int = integrate.quad(self.phi_bar, 0.0, t)[0]
        exp = np.exp(-(phi_int - self.gamma*t) / self.c_alpha)
        return self.x0_mean * exp

    def optimal_MFC_control(self, t, x):
        return -(self.eta(t)*x + (self.phi_bar(t) - self.eta(t) - self.gamma)*self.x_bar_MFC(t)) / self.c_alpha
