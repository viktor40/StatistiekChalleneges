"""
01/12/2020

Python Challenges Statistiek en Gegevensverwerking
Opdracht 2
Groep 4: Viktor Van Nieuwenhuize
         Aiko Decaluwe
         Fien Dewit

Indeling .py bestand:
    1.

"""

import numpy as np
import time
import matplotlib.pyplot as plt


"""----- Constanten -----"""
samples = np.loadtxt('./gamma_samples.dat')
M = 100
N_START = 10
N_MAX = 1000
STEPS = 10


def main():
    bootstrap_mm = repeat_bootstrap(methode_van_de_momenten_schatters)
    plot(bootstrap_mm)


def timer(func):
    """
    Een decorator om de performantie van een functie te testen. Deze functie neemt een andere functie als argument,
    voert de functie uit en print de tijd dat het duurde om deze uit te voeren.

    :param func: De functie waarvan we de performantie willen testen.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('Het duurde {}s om de functie "{}" uit te voeren'.format(end_time - start_time, func.__name__))
        return result
    return wrapper


def pick_random_samples(N):
    return np.random.choice(samples, size=N)


def covariantie_momenten_MM(xi, xi_2, gem_x, gem_x2, N):
    V_11 = 1 / (N * (N - 1)) * np.sum((xi - gem_x)**2)  # cov(k, k)
    V_12 = 1 / (N * (N - 1)) * np.sum((xi - gem_x) * (xi_2 - gem_x2))  # cov(k, theta)
    V_22 = 1 / (N * (N - 1)) * np.sum((xi_2 - gem_x2) ** 2)  # cov(theta, theta)
    return V_11, V_12, V_22


def afgeleides_MM_schatters(gem_x, gem_x2):
    dk_dx = (2 * gem_x * gem_x2) / (gem_x2 - gem_x ** 2)
    dk_dx2 = - gem_x2 ** 2 / (gem_x2 - gem_x ** 2) ** 2
    dt_dx = - (gem_x2 + gem_x ** 2) / gem_x ** 1
    dt_dx2 = 1 / gem_x
    return dk_dx, dk_dx2, dt_dx, dt_dx2


def covariantie_MM_schatters():
    xi = samples
    xi_2 = samples**2

    gem_x = np.average(xi)
    gem_x2 = np.average(samples)

    dk_dx, dk_dx2, dt_dx, dt_dx2 = afgeleides_MM_schatters(gem_x, gem_x2)
    cov_x_x, cov_x_x2, cov_x2_x2 = covariantie_momenten_MM(xi, xi_2, gem_x, gem_x2)

    var_k = dk_dx ** 2 * cov_x_x + dk_dx2 ** 2 + cov_x2_x2 + 2 * dk_dx * dk_dx2 * cov_x_x2
    var_theta = dt_dx ** 2 * cov_x_x + dt_dx2 ** 2 + cov_x2_x2 + 2 * dt_dx * dt_dx2 * cov_x_x2
    cov_k_theta = dk_dx * dt_dx * cov_x_x + dk_dx2 * dt_dx2 * cov_x2_x2 + (dk_dx * dt_dx2 + dt_dx * dt_dx2) * cov_x_x2
    cor_k_theta = cov_k_theta / np.sqrt(var_k * var_theta)

    return var_k, var_theta, cor_k_theta


def methode_van_de_momenten_schatters(data):
    gem_x = np.average(data)
    gem_x2 = np.average(data**2)

    k = gem_x**2 / (gem_x2 - gem_x ** 2)
    theta = (gem_x2 - gem_x ** 2) / gem_x
    return k, theta


echte_k, echte_theta = methode_van_de_momenten_schatters(samples)


def bias(k_bootstrap, theta_bootstrap):
    bias_k = np.average(k_bootstrap) - echte_k
    bias_theta = np.average(theta_bootstrap) - echte_theta

    return bias_k, bias_theta


def bootstrap(N, func):
    k_bootstrap, theta_bootstrap = np.array([func(pick_random_samples(N)) for _ in range(M)]).T

    bias_k, bias_theta = bias(k_bootstrap, theta_bootstrap)
    var_k = np.var(k_bootstrap)
    var_theta = np.var(theta_bootstrap)
    corr = np.corrcoef([k_bootstrap, theta_bootstrap])
    return bias_k, bias_theta, var_k, var_theta, corr[0][1]


@timer
def repeat_bootstrap(func):
    repeat_data = np.array([bootstrap(N, func) for N in range(N_START, N_MAX, STEPS)]).T
    return repeat_data


def plot(data):
    bias_k, bias_theta, var_k, var_theta, corr = data

    x_values = np.array(range(N_START, N_MAX, STEPS))
    plt.scatter(x_values, bias_k, c='red', marker='.', label='theta')
    plt.scatter(x_values, bias_theta, c='blue', marker='.', label='theta')
    plt.xlabel('N iteraties'), plt.ylabel('Bias')
    plt.title('Bias MM')
    plt.legend()
    plt.show()
    plt.clf()

    plt.scatter(x_values, var_k, c='red', marker='.', label='k')
    plt.scatter(x_values, var_theta, c='blue', marker='.', label='theta')
    plt.xlabel('N iteraties'), plt.ylabel('Variantie')
    plt.title('Variantie MM')
    plt.legend()
    plt.show()
    plt.clf()

    plt.scatter(x_values, corr, c='red', marker='.')
    plt.xlabel('N iteraties'), plt.ylabel('Correlatiecoëfficiënt')
    plt.title('Correlatiecoëfficiënt MM')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
