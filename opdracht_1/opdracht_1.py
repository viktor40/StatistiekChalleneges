import numpy as np
import matplotlib.pyplot as plt


def sferisch_to_cartesisch(array):
    r, theta, phi = array[0], array[1], array[2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


punten_sferisch = np.loadtxt("./punten_sferisch.dat")
punten_cartesisch =sferisch_to_cartesisch(np.transpose(punten_sferisch))
print(punten_cartesisch)

plt.hist(punten_cartesisch[0], bins=30, density=True, alpha=0.25, color='green', histtype='bar')
plt.hist(punten_cartesisch[0], bins=30, density=True, alpha=1, color='green', histtype='step')
plt.hist(punten_cartesisch[1], bins=30, density=True, alpha=0.25, color='red', histtype='bar')
plt.hist(punten_cartesisch[1], bins=30, density=True, alpha=1, color='red', histtype='step')
plt.hist(punten_cartesisch[2], bins=30, density=True, alpha=0.25, color='blue', histtype='bar')
plt.hist(punten_cartesisch[2], bins=30, density=True, alpha=1, color='blue', histtype='step')
plt.show()

