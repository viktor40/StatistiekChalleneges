import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def sferisch_to_cartesisch(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


# Deel 1
punten_sferisch = np.loadtxt('./punten_sferisch.dat')
data = np.transpose(punten_sferisch)
punten_cartesisch = sferisch_to_cartesisch(data[0], data[1], data[2])

plt.hist(punten_cartesisch[0], bins=30, density=True, alpha=0.25, color='green', histtype='bar')
plt.hist(punten_cartesisch[0], bins=30, density=True, alpha=1, color='green', histtype='step')
plt.hist(punten_cartesisch[1], bins=30, density=True, alpha=0.25, color='red', histtype='bar')
plt.hist(punten_cartesisch[1], bins=30, density=True, alpha=1, color='red', histtype='step')
plt.hist(punten_cartesisch[2], bins=30, density=True, alpha=0.25, color='blue', histtype='bar')
plt.hist(punten_cartesisch[2], bins=30, density=True, alpha=1, color='blue', histtype='step')
plt.legend(handles=[mpatches.Patch(color='green', label='x'),
                    mpatches.Patch(color='red', label='y'),
                    mpatches.Patch(color='blue', label='z')
                    ], loc='upper right')

plt.xlabel('Coordinaat'), plt.ylabel('Waarschijnlijkheidsdichtheid')
plt.title('Histogrammen van de cartesische coördinaten')
plt.show()


def afgeleide_r(r, theta, phi):
    dx_dr = np.sin(theta) * np.cos(phi)
    dy_dr = np.sin(theta) * np.sin(phi)
    dz_dr = np.cos(theta)
    return np.transpose(np.array([dx_dr, dy_dr, dz_dr]))


def afgeleide_theta(r, theta, phi):
    dx_dt = r * np.cos(theta) * np.cos(phi)
    dy_dt = r * np.cos(theta) * np.sin(phi)
    dz_dt = - r * np.sin(theta)
    return np.transpose(np.array([dx_dt, dy_dt, dz_dt]))


def afgeleide_phi(r, theta, phi):
    dx_dp = - r * np.sin(theta) * np.sin(phi)
    dy_dp = r * np.sin(theta) * np.cos(phi)
    dz_dp = 0
    return np.transpose(np.array([dx_dp, dy_dp, dz_dp]))


def matrix_vermenigvuldiging(r, theta, phi):
    SIGMA_R, SIGMA_THETA, SIGMA_PHI = 0.001, 0.003, 0.005
    CR = np.array([[SIGMA_R ** 2, 0, 0],
                   [0, SIGMA_THETA ** 2, 0],
                   [0, 0, SIGMA_PHI ** 2]])

    """
    A = [[dx/dr, dx/dt, dx/dp]
         [dy/dr, dy/dt, dy/dp]
         [dz/dr, dz/dt, dz/dp]]
    """

    A = np.array([afgeleide_r(r, theta, phi),
                  afgeleide_theta(r, theta, phi),
                  afgeleide_phi(r, theta, phi)])

    B = np.transpose(A)
    CX = np.dot(np.dot(A, B), CR)
    return CX


a = matrix_vermenigvuldiging(data[0], data[1], data[2])
print(a)
