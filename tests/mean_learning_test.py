import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

n_iters = 100_000
true_means = [1 + 0.5 / (1 + k)**(1/4) for k in range(n_iters)]


def learn_mean(omega):
    a = 1
    mean = 0.0
    mean_list = []
    for k in range(n_iters):
        rho = 1 / (a + k)**omega
        x = np.random.normal(true_means[k], 1.0)
        mean += rho * (x - mean)
        mean_list.append(mean)
    print(f'Learned mean for {omega=}: {mean}')
    return mean_list


if __name__ == '__main__':
    omegas = [0.6, 0.8, 1.0, 1.2, 1.4]
    df = pd.DataFrame()
    for omega in omegas:
        df[f'{omega=}'] = learn_mean(omega)
    df['true mean'] = true_means
    df.plot()
    plt.grid()
    plt.ylabel(r'$m$')
    plt.xlabel(r'$k$')
    plt.annotate(r'$\rho^m = \frac{1}{(100 + k)^\omega}$',
                 xy=(20000, 1.65), fontsize=20)
    plt.show()
