import sys

sys.path.append('../..')

import torch
import numpy as np
from infinite_horizon.LqIhEnv import LqIhEnv
from NNModels import ActorNet, CriticNet
from logger import Logger
from utils import flatten, get_params, save_actor_critic, get_policy_grad, plot_results
from tqdm import trange
from joblib import Parallel, delayed

beta = 1

ratio = 100  # time points in [0,1]

delta_t = 1 / ratio

gamma = np.exp(-beta * delta_t)

T = 20
nstep = int(T / delta_t)

l = 1.5
d = 1 / 4
kk = 0.6
xi = 1 / 2
c5 = 1

sigma = 0.3


def new_state(state, action, noise):
    env_sigma = 0.3

    # ratio=100 # time points in [0,1]

    delta_t = 1 / ratio

    new_state = state + action * delta_t + env_sigma * np.sqrt(delta_t) * noise

    return new_state


def reward_step(s, act, m):
    r_val = 0.5 * act**2 + d * (s - l * m)**2 + xi * (s - kk)**2 + c5 * m**2

    rew = r_val * delta_t

    return rew


def train_actor_critic(run_number, num_episodes, rho_V, rho_pi, omega):
    outdir = f'eps{episodes}_rhoV{rho_V}_rhopi{rho_pi}_omega{omega}_run{run_number}'
    omega_mu = np.round(omega, 2)

    critic = CriticNet(state_dim=1)
    actor = ActorNet(state_dim=1, action_dim=1)
    critic.double()
    actor.double()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=rho_V)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=rho_pi)

    sample_mean = np.zeros((nstep, 2))
    sample_std = np.ones(nstep)
    sample_M = np.zeros(nstep)
    log = Logger(
        'states',
        'state mean',
        'actions',
        'critic values',
        'deltas',
        # 'grad(log(pi))',
        # 'delta * grad(log(pi))',
        'action distribution std',
    )
    episodes_completed = 0

    try:

        print(f'Run {run_number}:')
        print('Training model...')

        for episode in trange(num_episodes + 1):

            s_sigma = sample_std[-1]
            x0 = np.random.normal(sample_mean[-1, 1], s_sigma)
            bound = 3 * s_sigma + sample_mean[-1, 1]
            if x0 > bound:
                x0 = bound
            elif x0 < -bound:
                x0 = -bound

            state = np.reshape(x0, (-1, 1))
            steps = 0
            z = np.random.normal(0, 1, size=nstep)

            while steps < nstep:
                l_mu = 1 / (1 + episode)**omega_mu

                # --Update mean field--
                sample_mean[steps, 1] = sample_mean[steps, 1] + l_mu * (state - sample_mean[steps, 1])

                # --Welford's algorithm to update standard deviation--
                sample_M[steps] = sample_M[steps] + (state - sample_mean[steps, 0]) * (state - sample_mean[steps, 1])
                if episode > 0:
                    sample_std[steps] = np.sqrt(sample_M[steps] / episode)

                state_tensor = torch.tensor(state)
                action_distribution = actor(state_tensor)
                action = action_distribution.sample()
                next_state = new_state(state, action.numpy(), z[steps])
                next_state_tensor = torch.tensor(next_state)
                reward = reward_step(state, action.numpy(), sample_mean[steps, 1])

                # --Update critic and actor--
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                with torch.no_grad():
                    V_of_next_state = critic(next_state_tensor)
                    target = torch.tensor(reward) + torch.tensor(gamma) * V_of_next_state
                critic_output = critic(state_tensor)
                delta = target - critic_output
                critic_loss = delta**2
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                log_prob = action_distribution.log_prob(action)
                actor_loss = -delta.detach() * log_prob
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # --Compute 2 norm of grad(delta * log(pi)) and grad(log(pi)--
                # grads = [p.grad.tolist() for p in actor.parameters()]
                # grads = flatten(grads)
                # grads_norm = sum([g ** 2 for g in grads]) ** (1 / 2)
                # norm_grad_log = get_policy_grad(action_distribution, action.item())

                log.log_data(
                    state.item(),
                    sample_mean[steps, 1].item(),
                    action.item(),
                    critic_output.item(),
                    delta.item(),
                    # norm_grad_log,
                    # grads_norm,
                    action_distribution.scale.item(),
                )

                state = next_state
                sample_mean[steps, 0] = sample_mean[steps, 1]
                steps += 1

            episodes_completed = episode

    except ValueError:
        print('Values are exploding.')
        print(f'Terminating learning after {episodes_completed} episodes')
    save_actor_critic(actor, critic, outdir)
    log.file_data(outdir)
    final_mean = log.log['state mean'][-1]
    plot_results(actor, final_mean, LqIhEnv(), episodes_completed, rho_V, rho_pi, omega, outdir)


if __name__ == '__main__':
    runs = [0, 1, 2, 3]
    episodes, rho_V, rho_pi, omega = get_params()
    Parallel(n_jobs=len(runs))(delayed(train_actor_critic)(n, episodes, rho_V, rho_pi, omega) for n in runs)
