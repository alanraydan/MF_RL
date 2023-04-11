import sys
sys.path.append('../..')

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from networks import CriticNet
import torch

CRITIC_PATH = '/finite_horizon/lq_control_problem/time_dependent/200000epssigma1.0_run0/critic.pt'

h = 0.0
m = 1.0
sigma = 1.0
c = 0.5
d = 0.5
r = 1.0
T = 1.0


def riccati(t):
    ts = np.linspace(T, t, 100)
    dt = ts[0] - ts[1]
    return integrate.odeint(lambda y, t: -2*h*y + (m*y)**2/d - c, r, ts), dt


def benchmark_value(t, x):
    ss, dt = riccati(t)
    s = ss[-1].item()
    integral_s = dt * sum(ss[1:] + ss[:-1])
    return s * x**2 + sigma**2 * integral_s


learned_value = CriticNet(state_dim=2)
learned_value.load_state_dict(torch.load(CRITIC_PATH))
learned_value.eval()

ts = [0.0, 0.5, 1.0]
xs = torch.linspace(-0.5, 1.5, 100).view(-1, 1)

fig, axs = plt.subplots(len(ts), 1, figsize=(6, 10))
for i, t in enumerate(ts):
    t_tensor = t * torch.ones_like(xs)
    tx = torch.hstack((t_tensor, xs))
    with torch.no_grad():
        axs[i].plot(xs, -learned_value(tx), label='learned value')
    axs[i].plot(xs, [benchmark_value(t, x.item()) for x in xs], label='benchmark value')
    axs[i].set_xlabel(f't = {t}')
    if i == 0:
        axs[i].legend()
    axs[i].grid()

fig.suptitle(r'Learned Value Function $V(t,x)$')
plt.tight_layout()

plt.show()
