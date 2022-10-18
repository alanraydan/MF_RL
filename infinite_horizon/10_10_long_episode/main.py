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


def train_actor_critic(n_steps, rho_V, rho_pi, omega, outdir):

    critic = CriticNet(state_dim=1)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=rho_V)

    actor = ActorNet(state_dim=1, action_dim=1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=rho_pi)

    log = Logger(
        'states',
        'learned mean',
        'actions',
        'action distribution std',
        'critic values',
        'deltas',
        # 'delta * grad(log(pi))',
    )

    state = init_dist.sample()
    state = state.unsqueeze(0).unsqueeze(0)
    mean = init_dist.mean

    try:

        print('Training model...')

        for t in trange(n_steps):

            rho_mean = 1 / (1 + t)**omega

            # --Update mean field--
            mean = mean + rho_mean * (state - mean)

            # --Sample action--
            action_distribution = actor(state)
            action = action_distribution.sample()

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

            # --Update actor--
            log_prob = action_distribution.log_prob(action)
            actor_loss = -delta.detach() * log_prob
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            log.log_data(
                state.item(),
                mean.item(),
                action.item(),
                action_distribution.scale.item(),
                critic_output.item(),
                delta.item(),
                # grads_norm,
            )

            state = next_state

    except ValueError:
        print('Values are exploding.')
        print(f'Terminating learning after {t} steps')
    # save_actor_critic(actor, critic, outdir)
    log.file_data(outdir)
    plot_results(actor, LqIhEnv(), t, rho_V, rho_pi, omega, sigma, outdir)


if __name__ == '__main__':
    runs = [0, 1, 2, 3]
    n_steps, rho_V, rho_pi, omega = get_params()
    outdir = f'{n_steps}steps_{omega}omega_{rho_V}critic_{rho_pi}actor'
    Parallel(n_jobs=len(runs))(delayed(train_actor_critic)(n_steps, rho_V, rho_pi, omega, outdir + f'_run{run}') for run in runs)
