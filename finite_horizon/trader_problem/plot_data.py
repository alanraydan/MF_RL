import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = '/Users/alanraydan/Development/mf_rl/finite_horizon/trader_problem/2023_03_09_control/200000eps_0.0omega_run5/data.csv'
df = pd.read_csv(PATH)
df_selection = df[-(4*17):]

times = df_selection['time step']
col_names = ['states', 'critic values', 'actions', 'action distribution std', 'action mean', 'deltas']
col_display_names = [r'$X_t$', r'$V(X_t)$', r'$A_t$', r'$\sigma(\pi(X_t))$', r'$\mathbb{E}[\theta_t]$', r'$\delta$']
fig, ax = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
for n, (col, col_display) in enumerate(zip(col_names, col_display_names)):
    ax[n % 3, n % 2].plot(times, df_selection[col])
    ax[n % 3, n % 2].set_ylabel(col_display, fontsize='xx-large')
    ax[n % 3, n % 2].grid()
plt.tight_layout()
plt.show()