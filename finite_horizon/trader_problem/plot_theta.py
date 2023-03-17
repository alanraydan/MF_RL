import numpy as np
import scipy.integrate as integrate
import pandas as pd
import matplotlib.pyplot as plt

PATH = '/Users/alanraydan/Development/mf_rl/finite_horizon/trader_problem/2023_03_09_control/200000eps_0.0omega_run5/data.csv'


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


def action_mean_mfg(t):
    return -eta_bar(t) * x_bar_mfg(t) / c_alpha


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


def action_mean_mfc(t):
    return -(phi_bar(t) - gamma) * x_bar_mfc(t) / c_alpha


df = pd.read_csv(PATH)
learned_thetas = df['action mean']
times = [0.0, 7/16, 15/16]

fig, axs = plt.subplots(len(times), 1, figsize=(8, 10))
for i, t in enumerate(times):
    learned_theta = learned_thetas[int(t * 16)::17].tolist()
    action_mean_mfg_t = [action_mean_mfg(t)] * len(learned_theta)
    action_mean_mfc_t = [action_mean_mfc(t)] * len(learned_theta)
    axs[i].plot(action_mean_mfg_t, label='MFG', linewidth=2)
    axs[i].plot(action_mean_mfc_t, label='MFC', linewidth=2)
    axs[i].plot(learned_theta, label=r'learned $\theta_t$', linewidth=2, linestyle='--')
    if i == 0:
        axs[i].legend()
    axs[i].grid()
    axs[i].set_xlabel(f't = {t}')
    axs[i].set_ylim([-2.5, 4])

fig.suptitle(r'Learned Control Mean $\theta_t$')
plt.tight_layout()
plt.savefig(f'{PATH[:-9]}/thetas.png')
plt.show()



