"""
Actor-critic algorithm for the time-dependent LQ control problem.

State dynamics:
    dX(t) = (h * X(t) + m * u(t)) * dt + sigma * dW(t)

Cost function:
    J = int_{0}^{T} (c * X(t)^2 + d * u(t)^2) dt + r * X(T)^2
"""

import sys
sys.path.append('../../..')
sys.path.append('..')

import torch
import numpy as np
from tqdm import trange
from joblib import Parallel, delayed

from networks import ActorNet, CriticNet
from logger import Logger
from utils import get_params, save_actor_critic

# Discrete time parameters
dt = 1/50
T = 1.0
times = torch.arange(0.0, T + dt, dt)

# LQ Coefficients
h = 0.0
m = 1.0
sigma = 1.0
c = 0.5
d = 0.5
r = 1.0
x0_mean = 1.0
x0_var = 0.5
param_dict = {
    'h': h, 
    'm': m, 
    'sigma': sigma, 
    'c': c, 
    'd': d, 
    'r': r, 
    'x0_mean': x0_mean, 
    'x0_var': x0_var, 
    'T': T
    }


def learn_control(n_episodes, run, rho_V, rho_pi, outdir):

    critic_list = torch.nn.ModuleList([CriticNet(state_dim=1) for _ in range(len(times))])
    actor_list = torch.nn.ModuleList([ActorNet(state_dim=1, action_dim=1) for _ in range(len(times))])
    critic_optimizer_list = [torch.optim.Adam(critic.parameters(), lr=rho_V) for critic in critic_list]
    actor_optimizer_list = [torch.optim.Adam(actor.parameters(), lr=rho_pi) for actor in actor_list]

    log = Logger(
        'states',
        'actions',
        'action distribution std',
        'critic values',
        'deltas',
    )

    print(f'Run {run}:')
    print('Training model...')

    for _ in trange(n_episodes):

        x = torch.normal(mean=x0_mean, std=np.sqrt(x0_var), size=(1,))

        for i, t in enumerate(times):

            # Sample action
            action_dist = actor_list[i](x)
            action = action_dist.sample()

            # Observe cost
            if i < len(times) - 1:
                cost = (c * x**2 + d * action**2) * dt
            else:
                cost = r * x**2
            reward = -cost

            # Observe next state
            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt))
            x_next = x + (h * x + m * action) * dt + sigma * dW

            # Compute TD error and update critic
            with torch.no_grad():
                v_next = critic_list[i + 1](x_next) if i < len(times) - 1 else 0.0
                target = reward + v_next
            critic_output = critic_list[i](x)
            delta = target - critic_output
            critic_loss = delta**2
            critic_optimizer_list[i].zero_grad()
            critic_loss.backward()
            critic_optimizer_list[i].step()

            # Update actor
            log_prob = action_dist.log_prob(action)
            actor_loss = -delta.detach() * log_prob
            actor_optimizer_list[i].zero_grad()
            actor_loss.backward()
            actor_optimizer_list[i].step()

            log.log_data(
                x.item(),
                action.item(),
                action_dist.scale.item(),
                critic_output.item(),
                delta.item(),
            )

            x = x_next

    # Save all actor and critic networks using pytorch
    print('Saving model...')
    save_actor_critic(actor_list, critic_list, outdir)
    log.file_data(outdir)
    # plot_control_and_state_distribution(param_dict, actor, outdir)


if __name__ == '__main__':
    runs = [0, 1, 2, 3, 4]
    n_episodes, rho_V, rho_pi, _ = get_params()
    outdir = f'{n_episodes}eps'
    Parallel(n_jobs=len(runs))(
        delayed(learn_control)(n_episodes, run, rho_V, rho_pi, outdir + f'_dt{dt}_run{run}') for run in runs
    )
