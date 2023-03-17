import sys
sys.path.append('../..')

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from networks import ActorNet
import torch

POLICY_PATH = '/Users/alanraydan/Development/mf_rl/finite_horizon/lq_control/800000eps_run4/actor.pt'

h = 0.0
m = 1.0
sigma = 0.5
c = 0.5
d = 0.5
r = 1.0
T = 1.0


def riccati(t):
    ts = np.linspace(T, t, 50)
    return integrate.odeint(lambda y, t: -2*h*y + (m*y)**2/d - c, r, ts)[-1].item()


def benchmark_control(t, x):
    s = riccati(t)
    return -x * s * m / d


learned_policy = ActorNet(state_dim=2, action_dim=1)
learned_policy.load_state_dict(torch.load(POLICY_PATH))
learned_policy.eval()

xs = torch.linspace(-1.5, 2.5, 100)
times = [0.0, 0.5, 1.0]

# controls_at_times = []
# std_at_times = []
# for t in times:
#     ts = t * torch.ones_like(xs)
#     tx_tensor = torch.stack((ts, xs), dim=1)
#     with torch.no_grad():
#         learned_control = learned_policy(tx_tensor).mean
#         learned_control_std = learned_policy(tx_tensor).scale
#         controls_at_times.append(learned_control.squeeze().numpy())
#         std_at_times.append(learned_control_std.squeeze().numpy())

xs = np.linspace(-1.5, 2.5, 100)

fig, axs = plt.subplots(len(times), 1, figsize=(6, 10))

for i, t in enumerate(times):
    axs[i].plot(xs, benchmark_control(t, xs), label='benchmark control', linewidth=2)
    # axs[i].plot(xs, controls_at_times[i], label='learned control', linewidth=2, linestyle='--', color='g')
    # axs[i].fill_between(xs, controls_at_times[i] - std_at_times[i], controls_at_times[i] + std_at_times[i], color='g', alpha=0.2)
    if i == 0:
        axs[i].legend()
    axs[i].grid()
    axs[i].set_xlabel(f't = {t}')
    # axs[i].set_ylim([-3, 6])

fig.suptitle(r'Learned Control $\alpha(t,x)$')
plt.tight_layout()

# plt.savefig(f'{POLICY_PATH[:-9]}/controls.png')
plt.show()
