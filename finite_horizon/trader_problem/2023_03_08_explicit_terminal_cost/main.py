import sys
sys.path.append('../../..')

import torch
import numpy as np
from networks import ActorNet, CriticNet
from tqdm import trange
from logger import Logger
from utils import get_params, save_actor_critic
from joblib import Parallel, delayed


# Discrete time parameters
dt = 1/16
T = 1.0
times = torch.arange(0.0, T + dt, dt)

# MF environment parameters
c_alpha = 1.0
c_x = 2.0
gamma = 1.75
c_g = 0.3
sigma = 0.5
init_dist = torch.distributions.normal.Normal(0.5, 0.3)

action_means = torch.zeros_like(times)

critic = CriticNet(state_dim=2)
actor = ActorNet(state_dim=2, action_dim=1)


def learn_mean_field(n_episodes, run, rho_V, rho_pi, omega, outdir):

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=rho_V)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=rho_pi)

    log = Logger(
        'states',
        'action mean',
        'actions',
        'action distribution std',
        'critic values',
        'deltas',
        'rho nu',
    )

    try:
        print(f'Run {run}:')
        print('Training model...')

        for ep in trange(n_episodes):

            rho_nu = 1 / (1 + ep)**omega
            x = init_dist.sample(sample_shape=(1, ))

            for i, t in enumerate(times):

                t = t.view((1, ))
                tx = torch.cat((t, x))

                # --Sample action--
                action_dist = actor(tx)
                action = action_dist.sample()

                # --Update action mean--
                action_means[i] = action_means[i] + rho_nu * (action - action_means[i])

                # --Observe reward and next state--
                if i < len(times) - 1:
                    cost = (
                        action**2 * c_alpha / 2
                        + x**2 * c_x / 2
                        - gamma * x * action_means[i]
                    ) * dt
                else:
                    cost = x**2 * c_g / 2.0
                reward = -cost

                dW = np.random.normal(loc=0.0, scale=np.sqrt(dt))
                x_next = x + action * dt + sigma * dW
                tx_next = torch.cat((t + dt, x_next))

                # --Update critic--
                with torch.no_grad():
                    if i == len(times) - 2:
                        v_next = -x_next**2 * c_g / 2.0
                    elif i == len(times) - 1:
                        v_next = 0.0
                    else:
                        v_next = critic(tx_next)
                    target = reward + v_next
                critic_output = critic(tx)
                delta = target - critic_output
                critic_loss = delta**2
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # --Update actor--
                log_prob = action_dist.log_prob(action)
                actor_loss = -delta.detach() * log_prob
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                log.log_data(
                    x.item(),
                    action_means[i].item(),
                    action.item(),
                    action_dist.scale.item(),
                    critic_output.item(),
                    delta.item(),
                    rho_nu,
                )

                x = x_next
    except ValueError:
        print('Values are exploding.')
        print(f'Terminating learning after {ep} episodes')

    save_actor_critic(actor, critic, outdir)
    log.file_data(outdir)


if __name__ == '__main__':
    runs = [0, 1, 2, 3, 4]
    n_episodes, rho_V, rho_pi, omega = get_params()
    outdir = f'{n_episodes}eps_{omega}omega'
    Parallel(n_jobs=len(runs))(
        delayed(learn_mean_field)(n_episodes, run, rho_V, rho_pi, omega, outdir + f'_run{run}') for run in runs
    )


