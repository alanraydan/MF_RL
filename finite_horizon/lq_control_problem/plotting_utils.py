import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch

def plot_control_and_state_distribution(problem_params, learned_policy, outdir):
    """
    Plots the control and state distribution under the learned policy against the benchmark control and state distribution.
    """
    h = problem_params['h']
    m = problem_params['m']
    sigma = problem_params['sigma']
    c = problem_params['c']
    d = problem_params['d']
    r = problem_params['r']
    x0_mean = problem_params['x0_mean']
    x0_var = problem_params['x0_var']
    T = problem_params['T']

    # Helper functions to compute benchmark control and benchmark state distribution
    def riccati(t):
        return integrate.odeint(lambda y, t: -2*h*y + (m*y)**2/d - c, r, [T, t])[-1].item()

    def benchmark_control(t, x):
        s = riccati(t)
        return -x * s * m / d

    def benchmark_state_mean(t):
        int_s = integrate.quad(riccati, 0, t)[0]
        exp_term = np.exp(h*t - int_s*m/d)
        return x0_mean * exp_term

    def benchmark_state_std(t):
        variance = integrate.odeint(lambda y, t: 2 * (h - riccati(t)*m/d) * y + sigma**2, x0_var, [0, t])[-1].item()
        return np.sqrt(variance)

    @torch.no_grad()
    def generate_state_distribution_samples(t, policy, num_samples=5000):
        """
        Generates `num_samples` samples from the state distribution at time `t` under the `policy`.
        Evolution of the state is approximated using Euler-Maruyama discretization.
        """
        num_time_steps = 100
        dt = t / num_time_steps

        if x0_var == 0:
            x = x0_mean * np.ones((num_samples, 1))
        else:
            x = np.random.normal(loc=x0_mean, scale=np.sqrt(x0_var), size=(num_samples, 1))
        t = np.zeros((num_samples, 1))
        for _ in range(num_time_steps):
            tx = torch.hstack((torch.tensor(t, dtype=torch.float), torch.tensor(x, dtype=torch.float)))
            action = policy(tx).sample().numpy()
            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(num_samples, 1))
            x += (h * x + m * action) * dt + sigma * dW
            t += dt

        return x
    
    # Choose upper and lower bounds for x axis such that the benchmark control is mostly supported for all times in `times`
    times = [0.0, 0.5, 1.0]
    x_lower_bound = min([benchmark_state_mean(t) - 4 * benchmark_state_std(t) for t in times])
    x_upper_bound = max([benchmark_state_mean(t) + 4 * benchmark_state_std(t) for t in times])
    xs = np.linspace(x_lower_bound, x_upper_bound, 100)

    # Choose upper and lower bounds for y axis so that the benchmark control is mostly supported for all times in `times`
    control_lower_bound = min([benchmark_control(t, x) for t in times for x in xs])
    control_upper_bound = max([benchmark_control(t, x) for t in times for x in xs])
    scale = (control_upper_bound - control_lower_bound) * 0.1
    control_lower_bound, control_upper_bound = control_lower_bound - scale, control_upper_bound + scale
    state_dist_lower_bound = 0.0
    state_dist_upper_bound = max([stats.norm.pdf(x, benchmark_state_mean(t), benchmark_state_std(t)) for t in times for x in xs
                                  if not np.isnan(stats.norm.pdf(x, benchmark_state_mean(t), benchmark_state_std(t)))]) * 1.1
    
    # Evaluate learned policy
    controls_at_times = []
    std_at_times = []
    for t in times:
        ts = t * torch.ones_like(torch.tensor(xs, dtype=torch.float))
        tx_tensor = torch.stack((ts, torch.tensor(xs, dtype=torch.float)), dim=1)
        with torch.no_grad():
            learned_control = learned_policy(tx_tensor).mean
            learned_control_std = learned_policy(tx_tensor).scale
            controls_at_times.append(learned_control.squeeze().numpy())
            std_at_times.append(learned_control_std.squeeze().numpy())
    
    # Set up figure for plotting
    fig, axs_control = plt.subplots(len(times), 1, figsize=(6, 8))
    axs_state = np.empty_like(axs_control)
    for i, ax in enumerate(axs_control):
        axs_state[i] = ax.twinx()
        ax.set_xlim([x_lower_bound, x_upper_bound])

    # Plot learned control and benchmark control
    for i, t in enumerate(times):
        axs_control[i].plot(xs, benchmark_control(t, xs), label='benchmark control', linewidth=2)
        axs_control[i].plot(xs, controls_at_times[i], label='learned control', linewidth=2, linestyle='--', color='g')
        axs_control[i].fill_between(xs, controls_at_times[i] - std_at_times[i], controls_at_times[i] + std_at_times[i], color='g', alpha=0.2)
        if i == 0:
            axs_control[i].legend()
        axs_control[i].set_xlabel(f't = {t}')
        axs_control[i].set_ylim([control_lower_bound, control_upper_bound])
        axs_control[i].set_ylabel(r'$\alpha(x)$')

        # Plot state distribution undr learned control and benchmark state distribution
        state_samples = generate_state_distribution_samples(t, learned_policy)
        axs_state[i].hist(state_samples.squeeze(), bins=40, density=True, color='silver')
        if np.isclose(benchmark_state_std(t), 0.0):
            axs_state[i].axvline(benchmark_state_mean(t), color='tab:orange', linewidth=2)
        else:
            axs_state[i].plot(xs, stats.norm.pdf(xs, benchmark_state_mean(t), benchmark_state_std(t)), color='tab:orange', linewidth=2)
        axs_state[i].set_ylim([state_dist_lower_bound, state_dist_upper_bound])
        axs_state[i].set_ylabel(r'$\mu$')
        axs_control[i].set_zorder(1)
        axs_control[i].patch.set_visible(False)

    fig.suptitle(r'Learned Control $\alpha(t,x)$')
    plt.tight_layout()

    plt.savefig(f'./{outdir}/controls.png')
    # plt.show()