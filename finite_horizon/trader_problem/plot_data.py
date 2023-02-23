import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = '/Users/alanraydan/Development/mf_rl/finite_horizon/trader_problem/100000eps_0.05omega_run0/data.csv'
df = pd.read_csv(PATH)
df_selection = df[:]

times = df_selection['time step']
col_names = ['states', 'critic values', 'actions', 'action distribution std', 'action mean', 'deltas']
col_display_names = [r'$X_t$', r'$V(X_t)$', r'$A_t$', r'$\sigma(\pi(X_t))$', r'$\nu_t$', r'$\delta$']
fig, ax = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
for n, (col, col_display) in enumerate(zip(col_names, col_display_names)):
    ax[n % 3, n % 2].plot(times, df_selection[col])
    # if col == 'states' or col == 'actions':
        # mean = mean_std_mfg[col][0]
        # std = mean_std_mfg[col][1]
        # ax[n % 3, n % 2].plot(times, mean * np.ones_like(times))
        # ax[n % 3, n % 2].plot(times, (mean + 3 * std) * np.ones_like(times), color='r')
        # ax[n % 3, n % 2].plot(times, (mean - 3 * std) * np.ones_like(times), color='r')
    # if col == 'state mean':
        # ax[n % 3, n % 2].plot(times, mean_std_mfg['states'][0] * np.ones_like(times))
    ax[n % 3, n % 2].set_ylabel(col_display, fontsize='xx-large')
    ax[n % 3, n % 2].grid()
plt.tight_layout()
plt.show()