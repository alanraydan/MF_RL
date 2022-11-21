import sys
sys.path.append('..')

import torch
from infinite_horizon.LqIhEnv import LqIhEnv
from networks import ActorNet, CriticNet
from logger import Logger
from utils import flatten, get_params, plot_results, is_exploding, \
    save_actor_critic, get_policy_grad, learned_policy_mse
from tqdm import trange
from joblib import Parallel, delayed


def train_actor_critic(run_number, episodes, rho_V, rho_pi, omega):
    # --Extract parameters from config file--
    episodes, rho_V, rho_pi, omega = get_params()
    outdir = f'eps{episodes}_rhoV{rho_V}_rhopi{rho_pi}_omega{omega}_run{run_number}'

    # --Initialize variables for training loop--
    env = LqIhEnv()
    env_time_steps = int(env.T / env.dt)
    discount = torch.exp(-env.beta * env.dt)
    critic = CriticNet(state_dim=1)
    actor = ActorNet(state_dim=1, action_dim=1)
    state_mean = torch.zeros((env_time_steps, 2))
    state_std = torch.ones(env_time_steps)
    sample_M = torch.zeros(env_time_steps)
    log = Logger(
        'states',
        'state mean',
        'actions',
        'critic values',
        'deltas',
        'grad(log(pi))',
        'delta * grad(log(pi))',
        'action distribution std',
        'mse benchmark control vs learned policy'
    )
    flag = True
    mse_tolerance = 0.05

    try:
        print('Training model...')
        for episode in trange(episodes):

            rho_mean = 1 / (1 + episode) ** omega
            x_init = torch.normal(state_mean[-1, 1], state_std[-1]).view(1, 1)
            state = env.reset(x_init)

            for t in range(env_time_steps):

                # --Sample action--
                action_distribution = actor(state)
                action = action_distribution.sample()
                mse = learned_policy_mse(actor, env.optimal_control_mfg)
                if (mse < mse_tolerance) and flag:
                    plot_results(actor, env, episode, rho_V, rho_pi, omega, outdir + 'mse_stop')
                    flag = False

                # --Update mean field--
                state_mean[t, [1, 0]] = state_mean[t, [0, 1]]
                state_mean[t, 1] = state_mean[t, 0] + rho_mean * (state - state_mean[t, 0])

                # --Observe next state and reward--
                next_state, reward, noise = env.step(action, state_mean[t, 1])

                # --Update critic--
                with torch.no_grad():
                    v_next = critic(next_state)
                    target = reward + discount * v_next
                critic_output = critic(state)
                delta = target - critic_output
                critic_loss = delta ** 2
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
                grads_norm = sum([g**2 for g in grads])**(1/2)
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
                    mse
                )

                state = next_state

                if is_exploding(state.item(), delta.item(), action.item(), grads_norm):
                    print('Values are exploding.')
                    print(f'Terminating learning after {episode} episodes.')
                    episodes = episode
                    raise OverflowError

            # --Welford's algorithm to update standard deviation--
            sample_M[:] = sample_M[:] + (state - state_mean[:, 0]) * (state - state_mean[:, 1])
            if episode > 0:
                lower_bound = torch.tensor([1e-3])
                state_std = torch.sqrt(torch.maximum(sample_M, lower_bound) / episode)

    except (OverflowError, ValueError):
        pass
    save_actor_critic(actor, critic, outdir)
    log.file_data(outdir)
    final_mean = log.log['state mean'][-1]
    plot_results(actor, env, episodes, rho_V, rho_pi, omega, outdir)


if __name__ == '__main__':
    n_jobs = 2
    episodes, rho_V, rho_pi, omega = get_params()
    Parallel(n_jobs=n_jobs)(delayed(train_actor_critic)(n, episodes, rho_V, rho_pi, omega) for n in range(n_jobs))
