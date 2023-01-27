import sys

sys.path.append('../..')

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from infinite_horizon.LqIhEnv import LqIhEnv
from networks import ActorNet, CriticNet
from logger import Logger
from utils import get_params, save_actor_critic, plot_results, compute_param_norm, plot_learned_control_and_distribution
from tqdm import trange
from joblib import Parallel, delayed

# Discrete time parameters
dt = 1e-2

# MF environment parameters
c1 = 0.25
c2 = 1.5
c3 = 0.5
c4 = 0.6
c5 = 1.0
beta = 1.0
sigma = 0.3
discount = np.exp(-beta * dt)
init_dist = torch.distributions.normal.Normal(0.0, 1.0)

dx = 0.1
bins = np.arange(-1.5, 2.5, dx)

# Logging constants
# x_eval = torch.linspace(0.8 - 2.6 * 0.234, 0.8 + 2.6 * 0.234, 100).view(-1, 1)  # 99% of states in this range for MFG
# action_df = pd.DataFrame(index=np.squeeze(x_eval.numpy()))
# mean_df = pd.DataFrame()


def train_actor_critic(n_steps, run, rho_V, rho_pi, omega, outdir):
    critic = CriticNet(state_dim=1)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=rho_V)

    actor = ActorNet(state_dim=1, action_dim=1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=rho_pi)

    log = Logger(
        'states',
        'state mean',
        'actions',
        'action distribution std',
        'critic values',
        'deltas',
        'grad delta^2',
        'delta * grad(log(pi))',
        'grad(V)^2',
        'rho m',
    )

    state = init_dist.sample()
    state = state.unsqueeze(0).unsqueeze(0)
    mean = init_dist.mean
    mu_discrete = np.ones(len(bins) + 1) / (len(bins) + 1)

    try:

        print(f'Run {run}:')
        print('Training model...')

        for t in trange(n_steps):

            rho_mean = 1 / (1 + t)**omega

            # --Update mean field and discrete learned distribution--
            mean = mean + rho_mean * (state - mean)
            idx = np.digitize(state.numpy(), bins)
            empirical_dist = np.zeros_like(mu_discrete)
            empirical_dist[idx] = 1.0
            mu_discrete = mu_discrete + 1/(1+t)**0.8 * (empirical_dist - mu_discrete)

            # --Sample action--
            action_distribution = actor(state)
            action = action_distribution.sample()

            # Log relevant values
            # if t % 1000 == 0:
            #     with torch.no_grad():
            #         a = actor(x_eval).mean.numpy()
            #         action_df[t] = np.squeeze(a)

            # --Observe reward and next state--
            cost = (
                0.5 * action**2
                + c1 * (state - c2 * mean)**2
                + c3 * (state - c4)**2
                + c5 * mean**2
            ) * dt
            reward = -cost

            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt))
            next_state = state + action * dt + sigma * dW
            if t < 200_000:
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
                mean.item(),
                action.item(),
                action_distribution.scale.item(),
                critic_output.item(),
                delta.item(),
                critic_grads_norm,
                actor_grads_norm,
                grad_V_squared,
                rho_mean,
            )

            state = next_state

    except ValueError:
        print('Values are exploding.')
        print(f'Terminating learning after {t} steps')
    save_actor_critic(actor, critic, outdir)
    log.file_data(outdir)
    # action_df.to_csv(outdir + '/actions.csv')
    plot_learned_control_and_distribution(actor, bins, dx, mu_discrete, LqIhEnv(), t, rho_V, rho_pi, omega, outdir)
    # plot_results(actor, LqIhEnv(), t, rho_V, rho_pi, omega, sigma, outdir)


if __name__ == '__main__':
    runs = [5]
    n_steps, rho_V, rho_pi, omega = get_params()
    outdir = f'{n_steps}steps_{omega}omega'
    Parallel(n_jobs=len(runs))(
        delayed(train_actor_critic)(n_steps, run, rho_V, rho_pi, omega, outdir + f'_run{run}') for run in runs
    )
