import sys
sys.path.append('../..')

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from networks import ActorNet
import torch

POLICY_PATH = '/Users/alanraydan/Development/mf_rl/finite_horizon/trader_problem/300000eps_0.05omega_run4/actor.pt'

c_alpha = 1.0
c_x = 2.0
gamma = 1.75
c_g = 0.3
sigma = 0.5
T = 1.0
x_bar_0 = 0.5


def eta_bar(t):
    B = 1/c_alpha
    C = c_x
    D = -gamma / (2*c_alpha)
    R = D**2 + B * C
    delta_plus = -D + np.sqrt(R)
    delta_minus = -D - np.sqrt(R)

    exp_term = np.exp((delta_plus - delta_minus) * (T - t))
    numer = -C * (exp_term - 1) - c_g * (delta_plus * exp_term - delta_minus)
    denom = delta_minus * exp_term - delta_plus - c_g * B * (exp_term - 1)

    return numer / denom


def x_bar_mfg(t):
    integral = integrate.quad(eta_bar, 0.0, t)[0]
    return x_bar_0 * np.exp(-integral / c_alpha)


def eta(t):
    root_term = np.sqrt(c_x / c_alpha)
    exp_term = np.exp(2 * root_term * (T - t))

    numer = c_alpha * root_term - c_g - (c_alpha * root_term + c_g) * exp_term
    denom = c_alpha * root_term - c_g + (c_alpha * root_term + c_g) * exp_term

    return -c_alpha * root_term * numer / denom


def optimal_control_mfg(t, x):
    return -(eta(t) * x + (eta_bar(t) - eta(t)) * x_bar_mfg(t)) / c_alpha


def phi_bar(t):
    R = 1/c_alpha
    b = R * (gamma**2 * R - c_x)
    a = 2*gamma*R
    c1 = (-a + np.sqrt(a**2 - 4*b)) / 2
    c2 = (-a - np.sqrt(a**2 - 4 * b)) / 2

    exp_term = np.exp((T - t) * (c2 - c1))
    numer = (c2 + R*c_g) * c1 * exp_term - c2*(c1 + R*c_g)
    denom = (c2 + R*c_g) * exp_term - (c1 + R*c_g)

    return numer / (denom * -R)


def x_bar_mfc(t):
    integral = integrate.quad(phi_bar, 0.0, t)[0]
    return x_bar_0 * np.exp(-(integral - gamma*t) / c_alpha)


def phi(t):
    return eta(t)


def optimal_control_mfc(t, x):
    return -(phi(t) * x + (phi_bar(t) - phi(t) - gamma) * x_bar_mfc(t)) / c_alpha


learned_policy = ActorNet(state_dim=2, action_dim=1)
learned_policy.load_state_dict(torch.load(POLICY_PATH))
learned_policy.eval()

xs = torch.linspace(-1.5, 1.5, 100)
times = [0.0, 7/16, 15/16]

controls_at_times = []
for t in times:
    ts = t * torch.ones_like(xs)
    tx_tensor = torch.stack((ts, xs), dim=1)
    with torch.no_grad():
        learned_control = learned_policy(tx_tensor).mean
        controls_at_times.append(learned_control.squeeze().numpy())

xs = np.linspace(-1.5, 1.5, 100)

fig, axs = plt.subplots(len(times), 1, figsize=(8, 10))

for i, t in enumerate(times):
    axs[i].plot(xs, optimal_control_mfg(t, xs), label='MFG', linewidth=2)
    axs[i].plot(xs, optimal_control_mfc(t, xs), label='MFC', linewidth=2)
    axs[i].plot(xs, controls_at_times[i], label='learned control', linewidth=2, linestyle='--')
    if i == 0:
        axs[i].legend()
    axs[i].grid()
    axs[i].set_xlabel(f't = {t}')

fig.suptitle(r'Learned Control for $\omega^{\nu} = 0.4$')
plt.tight_layout()
plt.show()
