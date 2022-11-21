import sys

sys.path.append('../..')

import torch
import numpy as np
from infinite_horizon.LqIhEnv import LqIhEnv
from networks import ActorNet, CriticNet
from logger import Logger
from utils import get_params, save_actor_critic, plot_results, compute_param_norm
from tqdm import trange
from joblib import Parallel, delayed

# Discrete time parameters
T = 20  # Infinity horizon truncated at T >> 0
dt = 1e-2
timesteps_per_ep = int(T / dt)

# MF environment parameters
c1 = 0.25
c2 = 1.5
c3 = 0.5
c4 = 0.6
c5 = 1.0
beta = 1.0
sigma = 0.3
discount = np.exp(-beta * dt)

dx = 0.1
bins = np.arange(-1.5, 2.5, dx)
mean = 0.8


def train_actor_critic(n_episodes, run, rho_V, rho_pi, outdir):
    # np.random.seed(1)
    # torch.manual_seed(1)

    log = Logger(
        'states',
        'actions',
        'action distribution std',
        'critic values',
        'deltas',
        'grad delta^2',
        'delta * grad(log(pi))',
        'grad(V)^2',
    )
    state_mean = np.zeros((timesteps_per_ep, 2))
    state_std = np.ones(timesteps_per_ep)
    sample_M = np.zeros(timesteps_per_ep)

    actor = ActorNet(state_dim=1, action_dim=1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=rho_pi)

    critic = CriticNet(state_dim=1)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=rho_V)

    try:

        print(f'Run {run}:')
        print('Training model...')

        for ep in trange(n_episodes):

            s_sigma = state_std[-1]
            x0 = torch.distributions.normal.Normal(state_mean[-1, 1], s_sigma).sample()
            state = x0.unsqueeze(0).unsqueeze(0)

            for t in range(timesteps_per_ep):

                state_mean[t, 1] = state_mean[t, 1] + (1 / (1 + ep)**(0.8)) * (state.item() - state_mean[t, 1])

                # --Welford's algorithm to update standard deviation--
                sample_M[t] = sample_M[t] + (state - state_mean[t, 0]) * (state.item() - state_mean[t, 1])
                if ep > 0:
                    state_std[t] = np.sqrt(sample_M[t] / ep)

                # --Sample action--
                action_distribution = actor(state)
                action = action_distribution.sample()

                cost = (
                    0.5 * action**2
                    + c1 * (state - c2 * mean)**2
                    + c3 * (state - c4)**2
                    + c5 * mean**2
                ) * dt
                reward = -cost

                dW = np.random.normal(loc=0.0, scale=np.sqrt(dt))
                next_state = state + action * dt + sigma * dW
                if ep < 200:
                    next_state = torch.clip(next_state, -5, 5)

                # --Compute 2-norm of grad(critic)--
                value = critic(state)
                critic_optimizer.zero_grad()
                value.backward()
                grad_V_squared = compute_param_norm(critic.parameters(), squared=True)

                # --Update critic--
                with torch.no_grad():
                    v_next = critic(next_state)
                    target = reward + discount * v_next
                critic_output = critic(state)
                delta = target - critic_output
                critic_loss = delta**2
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # --Compute 2 norm of grad(delta^2)--
                critic_grads_norm = compute_param_norm(critic.parameters())

                # --Update actor--
                log_prob = action_distribution.log_prob(action)
                actor_loss = -delta.detach() * log_prob
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # --Compute 2 norm of grad(delta * log(pi))--
                actor_grads_norm = compute_param_norm(actor.parameters())

                log.log_data(
                    state.item(),
                    action.item(),
                    action_distribution.scale.item(),
                    critic_output.item(),
                    delta.item(),
                    critic_grads_norm,
                    actor_grads_norm,
                    grad_V_squared,
                )

                state = next_state
                state_mean[t, 0] = state_mean[t, 1]

    except ValueError:
        print('Values are exploding...')
        print(f'Terminating learning after episode {ep}...')
    save_actor_critic(actor, critic, outdir)
    log.file_data(outdir)
    plot_results(actor, LqIhEnv(), ep, rho_V, rho_pi, None, outdir)


if __name__ == '__main__':
    runs = [3, 4]
    n_episodes, rho_V, rho_pi, _ = get_params()
    outdir = f'{n_episodes}eps'
    Parallel(n_jobs=len(runs))(
        delayed(train_actor_critic)(n_episodes, run, rho_V, rho_pi, outdir + f'_run{run}') for run in runs
    )
