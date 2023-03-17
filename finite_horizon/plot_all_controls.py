import sys
sys.path.append('../..')

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from networks import ActorNet
import torch

POLICY_PATH = '/Users/alanraydan/Development/mf_rl/finite_horizon/trader_problem/2023_03_09_control/200000eps_0.0omega_run0tanh/actor.pt'

learned_policy = ActorNet(state_dim=2, action_dim=1)
learned_policy.load_state_dict(torch.load(POLICY_PATH))
learned_policy.eval()

xs = torch.linspace(-1.5, 1.5, 100)
times = [0.0, 7/16, 15/16]

controls_at_times = []
std_at_times = []
for t in times:
    ts = t * torch.ones_like(xs)
    tx_tensor = torch.stack((ts, xs), dim=1)
    with torch.no_grad():
        learned_control = learned_policy(tx_tensor).mean
        learned_control_std = learned_policy(tx_tensor).scale
        controls_at_times.append(learned_control.squeeze().numpy())
        std_at_times.append(learned_control_std.squeeze().numpy())

xs = np.linspace(-1.5, 1.5, 100)

for i, t in enumerate(times):
    plt.plot(xs, controls_at_times[i])

plt.show()