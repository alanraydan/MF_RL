import sys
sys.path.append('../..')

from joblib import Parallel, delayed
import tqdm
import numpy as np
import torch

from networks import ActorNet, CriticNet
from utils import plot_results, get_params, save_actor_critic
from infinite_horizon.LqIhEnv import LqIhEnv
from logger import Logger

np.random.seed(1)
torch.manual_seed(1)

# MF environment parameters
c1 = 0.25
c2 = 1.5
c3 = 0.5
c4 = 0.6
c5 = 1.0
beta = 1.0
dt = 0.01
discount = np.exp(-beta * dt)
init_distribution = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
m = 0.8


def train_actor_critic(n_steps, actor_lr, critic_lr, sigma, outdir):

    log = Logger(
        'states',
        'actions',
        'action distribution std',
        'critic values',
        'deltas',
        'grad delta^2',
        'delta * grad(log(pi))',
    )

    state_dim = 1
    action_dim = 1

    actor = ActorNet(state_dim, action_dim)
    actor_optim = torch.optim.Adam(actor.parameters(), actor_lr)

    critic = CriticNet(state_dim)
    critic_optim = torch.optim.Adam(critic.parameters(), critic_lr)

    state = init_distribution.sample()
    state = state.unsqueeze(0).unsqueeze(0)

    try:
        for t in tqdm.trange(n_steps):

            action_dist = actor(state)
            action = action_dist.sample()

            cost = (
                0.5 * action**2
                + c1 * (state - c2 * m)**2
                + c3 * (state - c4)**2
                + c5 * m**2
            ) * dt
            reward = -cost

            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt))
            next_state = state + action * dt + sigma * dW

            critic_optim.zero_grad()
            with torch.no_grad():
                td_target = reward + discount * critic(next_state)
            value = critic(state)
            delta = td_target - value
            critic_loss = delta**2
            critic_loss.backward()
            critic_optim.step()
            critic_grad_norm = compute_param_norm(critic.parameters())

            actor_optim.zero_grad()
            log_policy = action_dist.log_prob(action)
            actor_loss = -delta.detach() * log_policy
            actor_loss.backward()
            actor_optim.step()
            actor_grad_norm = compute_param_norm(actor.parameters())

            log.log_data(
                state.item(),
                action.item(),
                action_dist.scale.item(),
                value.item(),
                delta.item(),
                critic_grad_norm,
                actor_grad_norm,
            )

            state = next_state

    except ValueError:
        print('Values are exploding...')
        print(f'Terminating learning after step {t}...')

    log.file_data(outdir)
    plot_results(actor, LqIhEnv(), n_steps, critic_lr, actor_lr, None, sigma, outdir)
    save_actor_critic(actor, critic, outdir)


def compute_param_norm(params):
    grads_norm = 0.0
    with torch.no_grad():
        for p in params:
            param_norm = p.grad.data.norm(2)
            grads_norm += param_norm.item()**2
    return grads_norm**0.5


if __name__ == '__main__':
    runs = [1]
    n_steps, critic_lr, actor_lr, _ = get_params()
    sigma = 0.8
    outdir = f'{n_steps}steps_{critic_lr}critic_{actor_lr}actor_{sigma}sigma'
    Parallel(n_jobs=len(runs))(delayed(train_actor_critic)(n_steps, actor_lr, critic_lr, sigma, outdir + f'_run{run}') for run in runs)
