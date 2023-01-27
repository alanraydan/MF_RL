import sys
sys.path.append('../..')

import torch
import numpy as np
from infinite_horizon.LqIhEnv import LqIhEnv
from networks import ActorNet, CriticNet
from logger import Logger
from utils import get_params, save_actor_critic, plot_results, compute_param_norm, plot_learned_control_and_distribution
from tqdm import trange
from joblib import Parallel, delayed

# MF environment parameters
c1 = 0.25
c2 = 1.5
c3 = 0.5
c4 = 0.6
c5 = 1.0
beta = 1.0
sigma = 0.3

# Discrete time parameters
T = 20  # Infinity horizon truncated at T >> 0
dt = 1e-2
timesteps_per_ep = int(T / dt)
discount = np.exp(-beta * dt)

dx = 0.1
bins = np.arange(-1.5, 2.5, dx)


def train_actor_critic(run_number, episodes, rho_V, rho_pi, omega, outdir):

    critic = CriticNet(state_dim=1)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=rho_V)

    actor = ActorNet(state_dim=1, action_dim=1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=rho_pi)

    state_mean = np.zeros((timesteps_per_ep, 2))
    state_std = np.ones(timesteps_per_ep)
    sample_M = np.zeros(timesteps_per_ep)
    mu_discrete = np.ones(len(bins) + 1) / (len(bins) + 1)
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

    try:

        print(f'Run {run_number}:')
        print('Training model...')

        for episode in trange(episodes):

            rho_mean = 1 / (1 + episode)**omega

            s_sigma = state_std[-1]
            x0 = np.random.normal(state_mean[-1, 1], s_sigma)
            state = torch.tensor([[x0]], dtype=torch.float)

            for t in range(timesteps_per_ep):

                # --Update mean field--
                state_mean[t, 1] = state_mean[t, 1] + rho_mean * (state - state_mean[t, 1])
                idx = np.digitize(state.numpy(), bins)
                empirical_dist = np.zeros_like(mu_discrete)
                empirical_dist[idx] = 1.0
                mu_discrete = mu_discrete + rho_mean * (empirical_dist - mu_discrete)

                # --Welford's algorithm to update standard deviation--
                sample_M[t] = sample_M[t] + (state - state_mean[t, 0]) * (state - state_mean[t, 1])
                if episode > 0:
                    state_std[t] = np.sqrt(sample_M[t] / episode)

                # --Sample action--
                action_distribution = actor(state)
                action = action_distribution.sample()

                # --Observe reward and next state--
                cost = (
                    0.5 * action**2
                    + c1 * (state - c2 * state_mean[t, 1])**2
                    + c3 * (state - c4)**2
                    + c5 * state_mean[t, 1]**2
                ) * dt
                reward = -cost

                dW = np.random.normal(loc=0.0, scale=np.sqrt(dt))
                next_state = state + action * dt + sigma * dW
                if episode < 200:
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
                # torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10)
                critic_optimizer.step()

                # --Compute 2 norm of grad(delta^2)--
                critic_grads_norm = compute_param_norm(critic.parameters())

                # --Update actor--
                log_prob = action_distribution.log_prob(action)
                actor_loss = -delta.detach() * log_prob
                actor_optimizer.zero_grad()
                actor_loss.backward()
                # torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10)
                actor_optimizer.step()

                # --Compute 2 norm of grad(delta * log(pi))--
                actor_grads_norm = compute_param_norm(actor.parameters())

                log.log_data(
                    state.item(),
                    state_mean[t, 1].item(),
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
                state_mean[t, 0] = state_mean[t, 1]

    except ValueError:
        print('Values are exploding...')
        print(f'Terminating learning after {episode} episodes')

    save_actor_critic(actor, critic, outdir)
    log.file_data(outdir)
    plot_learned_control_and_distribution(actor, bins, dx, mu_discrete, LqIhEnv(), episode, rho_V, rho_pi, omega, outdir)
    # plot_results(actor, LqIhEnv(), t, rho_V, rho_pi, omega, sigma, outdir)


if __name__ == '__main__':
    runs = [0, 1, 2]
    n_episodes, rho_V, rho_pi, omega = get_params()
    outdir = f'{n_episodes}eps_{omega}omega_{rho_V}critic_{rho_pi}actor'
    Parallel(n_jobs=len(runs))(delayed(train_actor_critic)(n, n_episodes, rho_V, rho_pi, omega, outdir + f'_run{n}') for n in runs)
