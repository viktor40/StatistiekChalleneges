import numpy as np
import matplotlib.pyplot as plt

n = 10000
x_beta = np.random.beta(a=2, b=2, size=n)
x_triangular = np.random.triangular(left=0, mode=0.75, right=1, size=n)
x_uniform = np.random.uniform(low=0.0, high=1.0, size=n)
x_exponential = np.random.exponential(scale=0.25, size=n)

n_bins = 50

plt.hist(x_beta, n_bins, alpha=0.25, color='orange', histtype='bar', density=True)
plt.hist(x_beta, n_bins, alpha=1.0, color='orange', histtype='step', density=True)
plt.hist(x_uniform, n_bins, alpha=0.25, color='green', histtype='bar', density=True)
plt.hist(x_uniform, n_bins, alpha=1.0, color='green', histtype='step', density=True)
plt.hist(x_triangular, n_bins, alpha=0.25, color='red', histtype='bar', density=True)
plt.hist(x_triangular, n_bins, alpha=1.0, color='red', histtype='step', density=True)
plt.hist(x_exponential, n_bins, alpha=0.25, color='magenta', histtype='bar', density=True)
plt.hist(x_exponential, n_bins, alpha=1.0, color='magenta', histtype='step', density=True)
plt.xlim(0, 2)
plt.ylim()
plt.show()
