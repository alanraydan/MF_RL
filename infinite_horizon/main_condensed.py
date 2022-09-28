import sys
sys.path.append('..')

import torch
from infinite_horizon.LqIhEnv import LqIhEnv
from NNModels import ActorNet, CriticNet
from logger import Logger
from utils import flatten, get_params, save_actor_critic, get_policy_grad, plot_results
from tqdm import trange
from joblib import Parallel, delayed

c1 = torch.tensor([[0.25]])
c2 = torch.tensor([[1.5]])
c3 = torch.tensor([[0.5]])
c4 = torch.tensor([[0.6]])
c5 = torch.tensor([[1.0]])
beta = torch.tensor([[1.0]])
sigma = torch.tensor([[0.3]])
T = 20  # Infinity horizon truncated at T >> 0
dt = torch.tensor([[1e-2]])
timesteps_per_ep = int(T / dt)
discount = torch.exp(-beta * dt)


def train_actor_critic(run_number, episodes, rho_V, rho_pi, omega):
    outdir = f'eps{episodes}_rhoV{rho_V}_rhopi{rho_pi}_omega{omega}_run{run_number}'

    critic = CriticNet(state_dim=1, lr=rho_V)
    actor = ActorNet(state_dim=1, action_dim=1, lr=rho_pi)

    state_mean = torch.zeros((timesteps_per_ep, 2))
    state_std = torch.ones(timesteps_per_ep)
    sample_M = torch.zeros(timesteps_per_ep)
    log = Logger(
        'states',
        'state mean',
        'actions',
        'critic values',
        'deltas',
        'grad(log(pi))',
        'delta * grad(log(pi))',
        'action distribution std',
    )
    episodes_completed = 0

    try:

        print(f'Run number {run_number}:')
        print('Training model...')

        for episode in trange(episodes):

            rho_mean = 1 / (1 + episode) ** omega
            state = torch.normal(state_mean[-1, 1], state_std[-1]).view(1, 1)
            bound = 3 * state_std[-1] + state_mean[-1, 1]

            if state > bound:
                state = bound.view(1, 1)
            elif state < -bound:
                state = -bound.view(1, 1)

            for t in range(timesteps_per_ep):

                # --Sample action--
                action_distribution = actor(state)
                action = action_distribution.sample()

                # --Update mean field--
                #state_mean[t, [1, 0]] = state_mean[t, [0, 1]]
                #state_mean[t, 1] = state_mean[t, 0] + rho_mean * (state - state_mean[t, 0])
                state_mean[t, 1] = state_mean[t, 1] + rho_mean * (state - state_mean[t, 1])

                # --Welford's algorithm to update standard deviation--
                sample_M[t] = sample_M[t] + (state - state_mean[t, 0]) * (state - state_mean[t, 1])
                if episode > 0:
                    lower_bound = torch.tensor([1e-4])
                    state_std = torch.sqrt(torch.maximum(sample_M, lower_bound) / episode)

                # --Observe cost and next state--
                with torch.no_grad():
                    cost = 0.5 * action**2\
                           + c1 * (state - c2*state_mean[t, 1])**2\
                           + c3 * (state - c4)**2\
                           + c5 * state_mean[t, 1]
                    reward = -cost

                    dw = torch.normal(mean=0.0, std=torch.sqrt(dt))
                    next_state = state + action * dt + sigma * dw

                # --Update critic--
                with torch.no_grad():
                    v_next = critic(next_state)
                    target = reward + discount * v_next
                critic_output = critic(state)
                delta = target - critic_output
                critic_loss = delta**2
                critic.zero_grad()
                critic_loss.backward()
                with torch.no_grad():
                    for p in critic.parameters():
                        sgd_update = p - rho_V * p.grad
                        p.copy_(sgd_update)

                # --Update actor--
                log_prob = action_distribution.log_prob(action)
                actor_loss = -delta.detach() * log_prob
                actor.zero_grad()
                actor_loss.backward()
                with torch.no_grad():
                    for p in actor.parameters():
                        sgd_update = p - rho_pi * p.grad
                        p.copy_(sgd_update)

                # --Compute 2 norm of grad(delta * log(pi)) and grad(log(pi)--
                grads = [p.grad.tolist() for p in actor.parameters()]
                grads = flatten(grads)
                grads_norm = sum([g ** 2 for g in grads]) ** (1 / 2)
                norm_grad_log = get_policy_grad(action_distribution, action.item())

                log.log_data(
                    state.item(),
                    state_mean[t, 1].item(),
                    action.item(),
                    critic_output.item(),
                    delta.item(),
                    norm_grad_log,
                    grads_norm,
                    action_distribution.scale.item(),
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
    plot_results(actor, final_mean, LqIhEnv(), episodes_completed, rho_V, rho_pi, omega, outdir)


if __name__ == '__main__':
    runs = ['test']
    episodes, rho_V, rho_pi, omega = get_params()
    Parallel(n_jobs=len(runs))(delayed(train_actor_critic)(n, episodes, rho_V, rho_pi, omega) for n in runs)
