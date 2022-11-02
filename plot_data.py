import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = '/Users/alanraydan/Development/mf_rl/infinite_horizon/10_25_clipstates/omega_0.05/3000000steps_0.05omega_run1/data.csv'
mean_std = {'states': (0.8, 0.234), 'actions': (0.0, 0.192551)}
omega = 1.0
rho_V = 1e-5

df = pd.read_csv(PATH)

df_selection = df[:]

# df_selection.plot(x='time step', figsize=(10, 12), subplots=True, grid=True)
times = df_selection['time step']
fig, ax = plt.subplots(len(df_selection.columns) - 1, figsize=(10, 12))
for n, (col_name, col_data) in enumerate(df_selection.iteritems()):
    if col_name != 'time step':
        ax[n-1].plot(times, col_data, label=col_name)
        if col_name == 'states' or col_name == 'actions':
            mean = mean_std[col_name][0]
            std = mean_std[col_name][1]
            ax[n-1].plot(times, mean * np.ones_like(times))
            ax[n-1].plot(times, (mean + 2 * std) * np.ones_like(times), color='r')
            ax[n-1].plot(times, (mean - 2 * std) * np.ones_like(times), color='r')
        if col_name == 'state mean':
            ax[n-1].plot(times, mean_std['states'][0] * np.ones_like(times))
        if col_name == 'rho m':
            ax[n-1].plot(times, df_selection['grad(V)^2'] * rho_V, label=r'$\rho^V (\nabla V)^2$', zorder=0)
        ax[n-1].grid()
        ax[n - 1].legend(loc='upper right')
plt.tight_layout()
plt.show()
