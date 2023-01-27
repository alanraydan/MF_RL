import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = '/Users/alanraydan/Development/mf_rl/infinite_horizon/2023_01_18_dynamic_clipping2/3000000steps_0.8omega_run2/data.csv'

mean_std_mfg = {'states': (0.8, 0.234), 'actions': (0.0, 0.192551)}
mean_std_mfc = {'states': (0.192, 0.18), 'actions': (0.0, 0.18)}  # Confirm these values
omega = 1.0
rho_V = 1e-5

df = pd.read_csv(PATH)

df_selection = df[:]

# df_selection.plot(x='time step', figsize=(10, 12), subplots=True, grid=True)
times = df_selection['time step']
# fig, ax = plt.subplots(len(df_selection.columns) - 1, figsize=(10, 12))
# for n, (col_name, col_data) in enumerate(df_selection.iteritems()):
#     if col_name != 'time step':
#         ax[n-1].plot(times, col_data, label=col_name)
#         if col_name == 'states' or col_name == 'actions':
#             mean = mean_std_mfg[col_name][0]
#             std = mean_std_mfg[col_name][1]
#             ax[n-1].plot(times, mean * np.ones_like(times))
#             ax[n-1].plot(times, (mean + 3 * std) * np.ones_like(times), color='r')
#             ax[n-1].plot(times, (mean - 3 * std) * np.ones_like(times), color='r')
#         if col_name == 'state mean':
#             ax[n-1].plot(times, mean_std_mfg['states'][0] * np.ones_like(times))
#         if col_name == 'rho m':
#             ax[n-1].plot(times, df_selection['grad(V)^2'] * rho_V, label=r'$\rho^V (\nabla V)^2$', zorder=0)
#         ax[n-1].grid()
#         ax[n - 1].legend(loc='upper right')
# plt.tight_layout()
# plt.show()

col_names = ['states', 'critic values', 'actions', 'action distribution std', 'state mean', 'deltas']
col_display_names = [r'$X_t$', r'$V(X_t)$', r'$A_t$', r'$\sigma(\pi(X_t))$', r'$m_t$', r'$\delta$']
fig, ax = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
for n, (col, col_display) in enumerate(zip(col_names, col_display_names)):
    ax[n % 3, n % 2].plot(times, df_selection[col])
    if col == 'states' or col == 'actions':
        mean = mean_std_mfg[col][0]
        std = mean_std_mfg[col][1]
        ax[n % 3, n % 2].plot(times, mean * np.ones_like(times))
        ax[n % 3, n % 2].plot(times, (mean + 3 * std) * np.ones_like(times), color='r')
        ax[n % 3, n % 2].plot(times, (mean - 3 * std) * np.ones_like(times), color='r')
    if col == 'state mean':
        ax[n % 3, n % 2].plot(times, mean_std_mfg['states'][0] * np.ones_like(times))
    ax[n % 3, n % 2].set_ylabel(col_display, fontsize='xx-large')
    ax[n % 3, n % 2].grid()
plt.tight_layout()
plt.show()
