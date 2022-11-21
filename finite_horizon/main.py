"""
Here we implement the U3-MF-AC algorithm for the Price Impact Model
to solve the extended MFG/MFC problems in continuous time. The MDP discretization of the
state dynamics and reward are implemented in PriceImpactEnv.py
"""


from finite_horizon.PriceImpactEnv import PriceImpactEnv
from networks import ActorNet, CriticNet
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse


def get_params():
    """
    Function for parsing parameters from config file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Enter parameter config filename")
    args = parser.parse_args()
    contents = open(args.filename).read()
    params = eval(contents)
    return args.filename, int(params['episodes']), params['critic_lr'],\
           params['actor_lr'], params['omega']


def convert_to_tensor(numpy_array):
    """
    Converts numpy arrays required for OpenAI's gym class into
    PyTorch tensors.
    """
    torch_tensor = torch.from_numpy(numpy_array).float()
    return torch.unsqueeze(torch_tensor, dim=0)


def plot_control(policy, env, filename=None):
    """
    Plots the learned control and optimal control for the given environment
    and stores the plots in a file named 'filename'.
    """
    x_vals = np.linspace(-1.5, 2, 100)
    t_vals = [0.0, 7 / 16, 15 / 16]
    fig, ax = plt.subplots(nrows=1, ncols=len(t_vals), figsize=(15, 6))
    for i, t in enumerate(t_vals):
        actions = np.zeros_like(x_vals)
        for j, x in enumerate(x_vals):
            state = torch.tensor([[t, x]]).float()
            with torch.no_grad():
                action_distribution = policy(state)
                action_mean = action_distribution.loc
            actions[j] = action_mean
        ax[i].plot(x_vals, actions,
                   x_vals, env.optimal_MFG_control(t, x_vals),
                   x_vals, env.optimal_MFC_control(t, x_vals))
        ax[i].set_xlabel('t={}'.format(t_vals[i]))
        ax[i].grid()
    if filename is not None:
        plt.savefig(filename.replace(".json", ".png"))
    else:
        plt.show()


if __name__ == '__main__':
    # Extract parameters from config file
    filename, episodes, rho_V, rho_pi, omega = get_params()
    env_time_steps = 100

    critic = CriticNet(state_dim=2)
    actor = ActorNet(state_dim=2, action_dim=1)
    control_mean = np.zeros(env_time_steps)

    for k in range(episodes):

        if k % 500 == 0:
            print(f"Beginning iteration number: {k} out of {episodes} ----------------------------------------------\n")

        env = PriceImpactEnv(control_distribution=control_mean)
        state = env.reset()
        state = convert_to_tensor(state)
        episode_done = False

        rho = 1/(1 + k)**omega

        while not episode_done:
            print(env.time_step)
            # Sample action
            action_distribution = actor(state)
            action = action_distribution.sample()

            # Observe next state and reward
            next_state, reward, episode_done, _ = env.step(action)
            next_state = convert_to_tensor(next_state)
            reward = torch.tensor(reward)

            # Update critic
            if not episode_done:
                target = reward + critic(next_state)
            else:
                target = reward
            critic.optimizer.zero_grad()
            critic_output = critic(state)
            delta = target.detach() - critic_output
            critic_loss = delta**2
            critic_loss.backward()
            critic.optimizer.step(state,, mf

            # Update actor
            actor.optimizer.zero_grad()
            log_prob = action_distribution.log_prob(action)
            actor_loss = -delta.detach()*log_prob
            actor_loss.backward()
            actor.optimizer.step(state,, mf

            # Update mean field
            # --indexing reflects that env has already advanced to the next timestep--
            # --no control at final timestep for MDP formulation--
            if not episode_done:
                control_mean[env.time_step - 1] = control_mean[env.time_step - 1]\
                                                  + rho*(action - control_mean[env.time_step - 1])
            state = next_state

    plot_control(actor, env, filename)
