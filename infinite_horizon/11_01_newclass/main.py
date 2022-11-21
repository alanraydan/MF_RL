import sys

sys.path.append('../..')

import torch
from tqdm import trange
from joblib import Parallel, delayed

from mf_env import IhMfEnv
from sde import ControlledSde
from running_cost_func import LqRunningCostFunc
from logger import Logger
from networks import ActorNet, CriticNet
from utils import get_params, save_actor_critic, plot_results

# Discrete time parameters
dt = 1e-2

# MF environment parameters
c1 = 0.25
c2 = 1.5
c3 = 0.5
c4 = 0.6
c5 = 1.0
beta = 1.0
init_dist = torch.distributions.normal.Normal(0.0, 1.0)

# mu(s, a, m) = a  and  sigma(s, a, m) = 0.3
sde = ControlledSde(mu=lambda s, a, m: a, sigma=lambda s, a, m: 0.3)
running_cost = LqRunningCostFunc(c1, c2, c3, c4, c4)


def learn_mean_field(n_steps, run, rho_V, rho_pi, omega, outdir):

    critic = CriticNet(state_dim=1)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=rho_V)
    actor = ActorNet(state_dim=1, action_dim=1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=rho_pi)

    env = IhMfEnv(init_dist, beta, running_cost, sde, dt)
    state = env.reset()
    state = state.unsqueeze(0).unsqueeze(0)
    mean = init_dist.mean

    try:

        print(f'Run {run}:')
        print('Training model...')

        for t in trange(n_steps):

            # --Update mean field--
            rho_mean = 1 / (1 + t)**omega
            mean = mean + rho_mean * (state - mean)

            # --Sample action--
            action_distribution = actor(state)
            action = action_distribution.sample()

            # --Observe cost and next state
            next_state, cost = env.step(state, action, torch.distributions.normal.Normal(mean, 1e-5))
            reward = -cost
            if t < 200_000:
                next_state = torch.clip(next_state, -5, 5)

            # --Update critic--
            with torch.no_grad():
                v_next = critic(next_state)
                target = reward + env.discount * v_next
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

            state = next_state

    except ValueError:
        print(f'Terminating learning after {t} steps')

    save_actor_critic(actor, critic, outdir)
    plot_results(actor, env, t, rho_V, rho_pi, omega, outdir)


if __name__ == '__main__':
    runs = [0, 1]
    n_steps, rho_V, rho_pi, omega = get_params()
    outdir = f'{n_steps}steps_{omega}omega'
    Parallel(n_jobs=len(runs))(
        delayed(learn_mean_field)(n_steps, run, rho_V, rho_pi, omega, outdir + f'_run{run}') for run in runs
    )
