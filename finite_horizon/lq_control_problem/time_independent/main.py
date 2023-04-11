import sys
sys.path.append('../../..')

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
sigma = 0.5
c = 1.0
d = 1.0
r = 1.0


def learn_control(n_episodes, run, rho_V, rho_pi, outdir):

    critic = CriticNet(state_dim=2)
    actor = ActorNet(state_dim=2, action_dim=1)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=rho_V)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=rho_pi)

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

        x = torch.tensor([1.0])

        for i, t in enumerate(times):

            t = t.view((1,))
            tx = torch.cat((t, x))

            # Sample action
            action_dist = actor(tx)
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
            tx_next = torch.cat((t + dt, x_next))

            # Compute TD error and update critic
            with torch.no_grad():
                v_next = critic(tx_next) if i < len(times) - 1 else 0.0
                target = reward + v_next
            critic_output = critic(tx)
            delta = target - critic_output
            critic_loss = delta**2
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Update actor
            log_prob = action_dist.log_prob(action)
            actor_loss = -delta.detach() * log_prob
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            log.log_data(
                x.item(),
                action.item(),
                action_dist.scale.item(),
                critic_output.item(),
                delta.item(),
            )

            x = x_next

    save_actor_critic(actor, critic, outdir)
    log.file_data(outdir)


if __name__ == '__main__':
    runs = [0, 1, 2, 3, 4]
    n_episodes, rho_V, rho_pi, _ = get_params()
    outdir = f'{n_episodes}eps'
    Parallel(n_jobs=len(runs))(
        delayed(learn_control)(n_episodes, run, rho_V, rho_pi, outdir + f'_run{run}') for run in runs
    )



