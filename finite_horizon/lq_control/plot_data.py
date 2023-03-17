import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = '/Users/alanraydan/Development/mf_rl/finite_horizon/lq_control/200000eps_run3/data.csv'
df = pd.read_csv(PATH)
df_selection = df[:]

times = df_selection['time step']
col_names = ['states', 'critic values', 'actions', 'action distribution std', 'deltas']
col_display_names = [r'$X_t$', r'$V(X_t)$', r'$A_t$', r'$\sigma(\pi(X_t))$', r'$\delta$']
fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
for n, (col, col_display) in enumerate(zip(col_names, col_display_names)):
    ax[n].plot(times, df_selection[col])
    ax[n].set_ylabel(col_display, fontsize='xx-large')
    ax[n].grid()
plt.tight_layout()
plt.show()
