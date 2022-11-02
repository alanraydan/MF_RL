import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LqIhEnv import LqIhEnv

env = LqIhEnv()
xs = np.linspace(0.8 - 2.6 * 0.234, 0.8 + 2.6 * 0.234, 100)
true_alphas = np.squeeze(env.optimal_control_mfg(xs).numpy())
true_mean = 0.8
actions_errors = pd.Series()

for i in range(10):
    df = pd.read_csv(f'/Users/alanraydan/Development/mf_rl/infinite_horizon/10_25_clipstates/3000000steps_0.8omega_run{i}/actions.csv', index_col=0)
    df = df.sub(true_alphas, axis=0)
    df = df**2
    df = df.sum()/len(xs)
    if i == 0:
        sums = df
    else:
        sums += df
sums /= 10

sums.plot(logy=True)
plt.xlabel('t')
plt.ylabel(r'MSE($\alpha$)')
plt.title(r'Error for $\alpha$ Averaged over 10 runs')
plt.grid()
plt.show()



