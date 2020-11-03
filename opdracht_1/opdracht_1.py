import numpy as np
import matplotlib.pyplot as plt


def sferisch_to_cartesisch(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def afgeleide_r(r, theta, phi):
    dx_dr = np.sin(theta) * np.cos(phi)
    dy_dr = np.sin(theta) * np.sin(phi)
    dz_dr = np.cos(theta)
    return dx_dr, dy_dr, dz_dr


def afgeleide_theta(r, theta, phi):
    dx_dt = r * np.cos(theta) * np.cos(phi)
    dy_dt = r * np.cos(theta) * np.sin(phi)
    dz_dt = - r * np.sin(theta)
    return dx_dt, dy_dt, dz_dt


def afgeleide_phi(r, theta, phi):
    dx_dp = - r * np.sin(theta) * np.sin(phi)
    dy_dp = r * np.sin(theta) * np.cosphi
    dz_dp = 0
    return dx_dp, dy_dp, dz_dp


punten_sferisch = np.loadtxt("./punten_sferisch.dat")
data = np.transpose(punten_sferisch)
punten_cartesisch = sferisch_to_cartesisch(data[0], data[1], data[2])

plt.hist(punten_cartesisch[0], bins=30, density=True, alpha=0.25, color='green', histtype='bar')
plt.hist(punten_cartesisch[0], bins=30, density=True, alpha=1, color='green', histtype='step')
plt.hist(punten_cartesisch[1], bins=30, density=True, alpha=0.25, color='red', histtype='bar')
plt.hist(punten_cartesisch[1], bins=30, density=True, alpha=1, color='red', histtype='step')
plt.hist(punten_cartesisch[2], bins=30, density=True, alpha=0.25, color='blue', histtype='bar')
plt.hist(punten_cartesisch[2], bins=30, density=True, alpha=1, color='blue', histtype='step')
plt.show()



