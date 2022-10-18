"""Set of helper functions for plotting, logging, and parsing command line args."""
import os
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import trange
import scipy.stats as stats
import numpy as np


def flatten(list_of_lists):
    """
    Recursive algorithm for flattening list of lists
    """
    if not list_of_lists:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def get_params():
    """
    Function for parsing parameters from config file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='Enter parameter config filename.')
    args = parser.parse_args()
    contents = open(args.infile).read()
    params = eval(contents)
    return int(params['episodes']), params['critic_lr'], params['actor_lr'], params['omega']


def plot_results(policy, env, episodes, critic_lr, actor_lr, omega, sigma, directory=None):
    """
    Plots the learned control and optimal control for the given environment, then generates and plots
    asymptotic samples of the states reached from following the learned control.

    The plots are saved in `./<directory>/results.png`.
    """
    x_vals = torch.linspace(-0.3, 1.8, 100).view(-1, 1)
    with torch.no_grad():
        action_mean = policy(x_vals).loc
    fig, ax1 = plt.subplots()
    ax1.plot(x_vals, action_mean, '--', linewidth=2, label='learned control', color='tab:blue')
    ax1.plot(x_vals, env.optimal_control_mfg(x_vals), linewidth=2, label='MFG control', color='tab:orange')
    ax1.plot(x_vals, env.optimal_control_mfc(x_vals), linewidth=2, label='MFC control', color='tab:green')
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$\alpha(x)$')
    ax1.legend()
    ax1.text(0.1, 0.1,
            f'eps = {episodes}\ncritic lr = {critic_lr}\nactor lr = {actor_lr}\nomega = {omega}',
            transform=ax1.transAxes)
    ax2 = ax1.twinx()
    asymptotic_state_samples = generate_asymptotic_samples(policy, sigma, 1000)
    ax2.hist(asymptotic_state_samples.view(1, -1), bins=40, density=True, color='silver')
    ax2.plot(x_vals, stats.norm.pdf(x_vals, 0.8, 0.234), color='tab:blue')
    ax2.set_ylabel(r'$\mu$')
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    if directory is not None:
        if not os.path.exists(f'./{directory}'):
            os.mkdir(f'./{directory}')
        plt.savefig(f'./{directory}/results.png')
    else:
        plt.show()


def generate_asymptotic_samples(policy, sigma, num_samples):
    """
    Returns a numpy array of samples from the asymptotic distribution for
    the given mean field game problem.
    """
    num_steps = 2000
    dt = 1e-2
    samples = torch.zeros(num_samples, 1)
    init_distribution = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
    print('Generating asymptotic samples...')
    for i in trange(num_samples):
        x = init_distribution.sample().view(1, 1)
        for t in range(num_steps):
            with torch.no_grad():
                action = policy(x).loc
                x = x + action * dt + sigma * np.random.normal(loc=0.0, scale=np.sqrt(dt))
        samples[i] = x
    return samples


def plot_data(step_range, data_dict, directory=None):
    lower = step_range[0]
    upper = step_range[1]
    print(lower, upper)
    assert lower >= 0 and upper <= len(data_dict['states'])
    steps = [i for i in range(lower, upper)]
    fig, axs = plt.subplots(len(data_dict), 1, figsize=(10, 10))
    for i, (key, val) in enumerate(data_dict.items()):
        axs[i].plot(steps, val[lower:upper])
        axs[i].set_title(f'{key}')
        axs[i].grid()
    fig.tight_layout()
    if directory is not None:
        if not os.path.exists(f'./{directory}'):
            os.mkdir(f'./{directory}')
        plt.savefig(f'./{directory}/{directory}_data.png')
    else:
        plt.show()


def is_exploding(*data):
    return any([abs(d) > 10e3 for d in data])


def save_actor_critic(actor, critic, directory):
    if not os.path.exists(f'./{directory}'):
        os.mkdir(f'./{directory}')
    torch.save(actor.state_dict(), f'./{directory}/actor.pt')
    torch.save(critic.state_dict(), f'./{directory}/critic.pt')


def get_policy_grad(distribution, value):
    mu = distribution.loc.item()
    sigma = distribution.scale.item() ** 2
    dlog_mu = (value - mu) * sigma ** -1
    dlog_sigma = 0.5 * (sigma ** -2 * (value - mu) ** 2 - sigma ** -1)
    return (dlog_mu ** 2 + dlog_sigma ** 2) ** (1 / 2)


def learned_policy_mse(policy, benchmark):
    x_vals = torch.linspace(-1.5, 3, 100).view(-1, 1)
    with torch.no_grad():
        action_mean = policy(x_vals).loc
    optimal_action = benchmark(x_vals)
    mse = sum((action_mean - optimal_action)**2) / len(x_vals)
    return mse.item()


def compute_param_norm(params, squared=False):
    grads_norm = 0.0
    with torch.no_grad():
        for p in params:
            param_norm = p.grad.data.norm(2)
            grads_norm += param_norm.item()**2
    return grads_norm if squared else grads_norm**0.5
