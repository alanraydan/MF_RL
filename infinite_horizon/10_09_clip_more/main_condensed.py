import sys
sys.path.append('../..')

import torch
import numpy as np
from infinite_horizon.LqIhEnv import LqIhEnv
from networks import ActorNet, CriticNet
from logger import Logger
from utils import get_params, save_actor_critic, plot_results
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


def train_actor_critic(run_number, episodes, rho_V, rho_pi, omega):

    outdir = f'eps{episodes}_rhoV{rho_V}_rhopi{rho_pi}_omega{omega}_run{run_number}'

    critic = CriticNet(state_dim=1)
    actor = ActorNet(state_dim=1, action_dim=1)
    # critic.double()
    # actor.double()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=rho_V)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=rho_pi)

    state_mean = np.zeros((timesteps_per_ep, 2))
    state_std = np.ones(timesteps_per_ep)
    sample_M = np.zeros(timesteps_per_ep)
    log = Logger(
        'states',
        'state mean',
        'actions',
        'action distribution std',
        'critic values',
        'deltas',
        'grad delta^2',
        'delta * grad(log(pi))',
    )
    episodes_completed = 0

    try:

        print(f'Run {run_number}:')
        print('Training model...')

        for episode in trange(episodes):

            rho_mean = 1 / (1 + episode)**omega

            s_sigma = state_std[-1]
            x0 = np.random.normal(state_mean[-1, 1], s_sigma)
            bound = 3 * s_sigma + state_mean[-1, 1]
            if x0 > bound:
                x0 = bound
            elif x0 < -bound:
                x0 = -bound

            state = np.reshape(x0, (-1, 1))

            for t in range(timesteps_per_ep):

                # --Update mean field--
                state_mean[t, 1] = state_mean[t, 1] + rho_mean * (state - state_mean[t, 1])

                # --Welford's algorithm to update standard deviation--
                sample_M[t] = sample_M[t] + (state - state_mean[t, 0]) * (state - state_mean[t, 1])
                if episode > 0:
                    state_std[t] = np.sqrt(sample_M[t] / episode)

                # --Sample action--
                state_tensor = torch.tensor(state, dtype=torch.float)
                action_distribution = actor(state_tensor)
                action_tensor = action_distribution.sample()
                action = action_tensor.numpy()

                # --Observe reward and next state--
                cost = (
                    0.5 * action**2
                    + c1 * (state - c2 * state_mean[t, 1])**2
                    + c3 * (state - c4)**2
                    + c5 * state_mean[t, 1]**2
                ) * dt
                reward = -cost

                noise = np.random.normal(loc=0.0, scale=1.0)
                next_state = state + action * dt + sigma * np.sqrt(dt) * noise
                next_state_tensor = torch.tensor(next_state, dtype=torch.float)

                # --Update critic--
                with torch.no_grad():
                    v_next = critic(next_state_tensor)
                    target = torch.tensor(reward) + torch.tensor(discount) * v_next
                critic_output = critic(state_tensor)
                delta = target - critic_output
                critic_loss = delta**2
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10)
                critic_optimizer.step()

                # --Compute 2 norm of grad(delta^2)--
                critic_grads_norm = 0.0
                with torch.no_grad():
                    for p in critic.parameters():
                        param_norm = p.grad.data.norm(2)
                        critic_grads_norm += param_norm.item()**2
                critic_grads_norm = critic_grads_norm**0.5


                # --Update actor--
                log_prob = action_distribution.log_prob(action_tensor)
                actor_loss = -delta.detach() * log_prob
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10)
                actor_optimizer.step()

                # --Compute 2 norm of grad(delta * log(pi))--
                actor_grads_norm = 0.0
                with torch.no_grad():
                    for p in actor.parameters():
                        param_norm = p.grad.data.norm(2)
                        actor_grads_norm += param_norm.item()**2
                actor_grads_norm = actor_grads_norm**0.5

                log.log_data(
                    state.item(),
                    state_mean[t, 1].item(),
                    action.item(),
                    action_distribution.scale.item(),
                    critic_output.item(),
                    delta.item(),
                    critic_grads_norm,
                    actor_grads_norm,
                )

                state = next_state
                state_mean[t, 0] = state_mean[t, 1]

            episodes_completed = episode

    except ValueError:
        print('Values are exploding.')
        print(f'Terminating learning after {episodes_completed} episodes')
    save_actor_critic(actor, critic, outdir)
    log.file_data(outdir)
    final_mean = log.log['state mean'][-1]
    plot_results(actor, LqIhEnv(), episodes_completed, rho_V, rho_pi, omega, sigma, outdir)


if __name__ == '__main__':
    runs = [9]
    episodes, rho_V, rho_pi, omega = get_params()
    Parallel(n_jobs=len(runs))(delayed(train_actor_critic)(n, episodes, rho_V, rho_pi, omega) for n in runs)
