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
from utils import get_params, save_actor_critic, plot_results, compute_param_norm

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

# Drift and volatility coefficients for SDE
# mu(s, a, m) = a  and  sigma(s, a, m) = 0.3
SIGMA = 0.3
sde = ControlledSde(mu=lambda s, a, m: a, sigma=lambda s, a, m: SIGMA)
running_cost = LqRunningCostFunc(c1, c2, c3, c4, c4)

CLIP_RANGE = 5
BOUNDARY_THRESHOLD = 5_000


def learn_mean_field(n_steps, run, rho_V, rho_pi, omega, outdir):

    critic = CriticNet(state_dim=1)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=rho_V)
    actor = ActorNet(state_dim=1, action_dim=1)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=rho_pi)

    env = IhMfEnv(init_dist, beta, running_cost, sde, dt)
    state = env.reset()
    state = state.unsqueeze(0).unsqueeze(0)
    mean = init_dist.mean
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

    boundary_counter = 0
    upper_clip = state + CLIP_RANGE
    lower_clip = state - CLIP_RANGE

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

            # Clip depending on mean with fixed `CLIP_RANGE`
            if next_state > upper_clip or next_state < lower_clip:
                boundary_counter += 1
            if boundary_counter > BOUNDARY_THRESHOLD:
                upper_clip = mean + CLIP_RANGE
                lower_clip = mean - CLIP_RANGE
            next_state = torch.clip(next_state, upper_clip, lower_clip)

            # --Compute 2-norm of grad(critic)--
            value = critic(state)
            critic_optimizer.zero_grad()
            value.backward()
            grad_V_squared = compute_param_norm(critic.parameters(), squared=True)

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
        print(f'Terminating learning after {t} steps')

    save_actor_critic(actor, critic, outdir)
    log.file_data(outdir)
    plot_results(actor, env, t, rho_V, rho_pi, omega, outdir)


if __name__ == '__main__':
    runs = [0, 1, 2, 3]
    n_steps, rho_V, rho_pi, omega = get_params()
    outdir = f'{n_steps}steps_{omega}omega'
    Parallel(n_jobs=len(runs))(
        delayed(learn_mean_field)(n_steps, run, rho_V, rho_pi, omega, outdir + f'_run{run}') for run in runs
    )
