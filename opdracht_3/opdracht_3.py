"""
15/12/2020

Python Challenges Statistiek en Gegevensverwerking
Opdracht 3
Groep 4: Viktor Van Nieuwenhuize
         Aiko Decaluwe
         Fien Dewit

Indeling .py bestand:
    1. constanten
    2. config
    3. main function
    4. timing decorator
    5. hit or miss methode
    6. monte carlo
    7. toevalsgetallen
    8. run het script
"""

import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time

"-----Constanten-----"
SIZE = 100000
N = 1000000


"-----Config-----"
rcParams.update({'font.size': 11})

HM_PUNTEN_PLOT = True
HM_HIST = False
HM_INV_CUM = False

MONTE_CARLO = False
STRAT = False
INT_FOUT = False

TOEVALSGETALLEN = True


def main():
    """
    In de main functie runnen we het script. Er kan in de config worden gekozen welke delen van het script moeten
    worden gerund.
    Voor de toevalsgetallen worden de functies uitgevoerd van 10E4 tot 10E8 zodat er een mooie evolutie te zien is.
    """
    u_samples = np.random.uniform(low=0.0, high=1.0, size=SIZE)

    if HM_PUNTEN_PLOT:
        plot_hit_or_miss_punten('uniform')
        plot_hit_or_miss_punten('triangulair')

    if HM_HIST:
        plot_hit_or_miss_hist('uniform')
        plot_hit_or_miss_hist('triangulair')

    if HM_INV_CUM:
        inv_cum()

    if MONTE_CARLO:
        u1 = montecarlo(1, 100)
        u2 = montecarlo(2, 100)
        print(u1, u2)

    if STRAT:
        u1 = stratificatie(1, 100)
        u2 = stratificatie(2, 100)
        print(u1, u2)

    if INT_FOUT:
        plot_integratie_fout(1, steps=2)
        plot_integratie_fout(2, steps=2)

    if TOEVALSGETALLEN:
        n = int(10E8)
        global N
        while n <= 10E8:
            N = n
            plot_2d_hist()
            plot_2d_doorsnede()
            n *= 10


def timer(func):
    """
    Een decorator om de performantie van een functie te testen. Deze functie neemt een andere functie als argument,
    voert de functie uit en print de tijd dat het duurde om deze uit te voeren. We printen er ook de parameters om
    zeker te zijn welke methode er werd gebruikt.

    :param func: De functie waarvan we de performantie willen testen.
    """

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        param = ''
        if args:
            for arg in args:
                param += str(arg) + ', '

        print(('Het duurde {}s om de functie "{}" uit te voeren. '
               'De functie had de argumenten: "{}"').format(end_time - start_time, func.__name__,
                                                            param[:-2] if param else None))
        return result
    return wrapper


"""-----Hit Or Miss-----"""


def f(x):
    """
    De functie uit de opgave
    :param x: Een array met de getrokken waarden
    """
    return np.pi * x * np.cos((np.pi / 2) * x ** 2)


def f_inv_cum(x):
    """
    De inverse cummulatieve van de functie uit de opgave
    :param x: Een array met de getrokken waarden
    """
    return np.sqrt((2 / np.pi) * np.arcsin(x))


def hx_triangular(x_samples):
    """
    Een functie om hx te helpen berekenen voor een triangulaire verdeling.
    :param x_samples: Een array met de getrokken waarden
    """
    hx = np.array([3.2 * x if x <= 0.75 else -9.6 * x + 9.6 for x in x_samples])
    return hx


@timer
def hit_or_miss(verdeling):
    """
    Deze functie past de hit or miss methode toe. Eerst worden er samples getrokken afhankelijk van de verdeling.
    Erna wordt er gekeken of het punt een hit of een miss is. Ten slotte wordt de efficientie nog berekend
    :param verdeling: Het type verdeling die we gebruiken voor de hit or miss. Dit is uniform of triangulair.
    :return: de x en y coordinaten voor de hits en misses
    """
    # random samples trekken
    u_samples = np.random.uniform(low=0.0, high=1.0, size=SIZE)
    if verdeling == 'uniform':
        x_samples = np.random.uniform(low=0.0, high=1.0, size=SIZE)
        y_samples = 1.61 * u_samples  # y_max = maximum van de functie in het interval [0, 1]
    elif verdeling == 'triangulair':
        x_samples = np.random.triangular(left=0, mode=0.75, right=1, size=SIZE)
        y_samples = hx_triangular(x_samples) * u_samples
    else:
        raise Exception('Ongekende verdeling')

    # pas de functie toe
    y_functie = f(x_samples)

    y_hit, x_hit, x_miss, y_miss = [], [], [], []
    for n in range(len(x_samples)):
        if y_samples[n] <= y_functie[n]:
            x_hit.append(x_samples[n]), y_hit.append(y_samples[n])
        else:
            x_miss.append(x_samples[n]), y_miss.append(y_samples[n])

    # efficientie
    print('efficientie = ', len(x_hit) / SIZE)
    return x_hit, y_hit, x_miss, y_miss


def plot_hit_or_miss_punten(verdeling):
    """
    Deze functie plot een voorstelling van de hit or miss methode, waarbij de hits en misses zijn geplot alsook de
    functie.
    :param verdeling: Het type verdeling die we gebruiken voor de hit or miss. Dit is uniform of triangulair.
    """
    x_hit, y_hit, x_miss, y_miss = hit_or_miss(verdeling)
    plt.plot(x_hit, y_hit, ',', label='x', c='g')
    plt.plot(x_miss, y_miss, ',', label='x', c='r')
    x = np.arange(0.0, 1.0, 0.001)
    plt.plot(x, f(x), linewidth=2.5, c='b')
    plt.title('Hit or Miss {}'.format(verdeling))
    plt.xlabel('x'), plt.ylabel('f(x)')
    plt.grid()
    plt.legend(handles=[mpatches.Patch(color='green', label='Hit'),
                        mpatches.Patch(color='red', label='Miss'),
                        mpatches.Patch(color='blue', label='f(x)')], loc='upper right')
    plt.show()
    plt.savefig('./plots/hit_or_miss/punten_{}.pdf'.format(verdeling))
    plt.clf()


def plot_hit_or_miss_hist(verdeling):
    """
    Deze functie plot een histogram van de hit or miss methode
    :param verdeling: Het type verdeling die we gebruiken voor de hit or miss. Dit is uniform of triangulair.
    """
    x_hit, y_hit, x_miss, y_miss = hit_or_miss(verdeling)
    plt.hist(x_hit, bins=50, density=True, color='tab:orange')
    x = np.arange(0.0, 1.0, 0.001)
    plt.plot(x, f(x), linewidth=2.5, c='b')
    plt.title('Hit or Miss {} histogram'.format(verdeling, ))
    plt.xlabel('x'), plt.ylabel('f(x)')
    plt.grid()
    plt.legend(handles=[mpatches.Patch(color='orange', label='Histogram'),
                        mpatches.Patch(color='blue', label='f(x)')], loc='upper right')
    plt.savefig('./plots/hit_or_miss/histogram_{}.pdf'.format(verdeling))
    plt.clf()


@timer
def inv_cum():
    u_samples = np.random.uniform(low=0.0, high=1.0, size=SIZE)
    x_samples = f_inv_cum(u_samples)
    plt.title('Inverse Cummulatieve'), plt.ylabel('f(x)'), plt.xlabel('x')
    plt.legend(handles=[mpatches.Patch(color='orange', label='Histogram'),
                        mpatches.Patch(color='blue', label='f(x)')], loc='upper right')
    plt.hist(x_samples, bins=50, density=True, color='tab:orange')

    t = np.arange(0.0, 1.0, 0.001)
    plt.plot(t, f(t), linewidth=2.5, c='b')
    plt.savefig('./plots/hit_or_miss/inv_cum.pdf')
    plt.clf()


"""-----Montecarlo-----"""


def montecarlo(b, n):
    xrand = np.array([np.random.uniform(0, b) for _ in range(n)])
    integral = sum(f(xrand))
    return b / float(n) * integral


def stratificatie(b, n):
    xrand = np.array([np.random.uniform(0, b / 2) if i % 2 == 0
                      else np.random.uniform(b / 2, b)
                      for i in range(n)])
    integral = sum(f(xrand))
    return b / float(n) * integral


def fout_integratie(b, n, methode):
    if methode == 'montecarlo':
        data = np.array([montecarlo(b, n) for _ in range(100)])
    elif methode == 'stratificatie':
        data = np.array([stratificatie(b, n) for _ in range(100)])
    else:
        raise Exception('Verkeerde Integratiemethode')
    return float(np.std(data))


@timer
def plot_integratie_fout(b, steps):
    i = 4
    samples_n = np.array([])
    fouten_mt, fouten_strat = np.array([]), np.array([])
    while i < 40000:
        i *= steps
        samples_n = np.append(samples_n, i)
        fouten_mt = np.append(fouten_mt, fout_integratie(b, i, 'montecarlo'))
        fouten_strat = np.append(fouten_strat, fout_integratie(b, i, 'stratificatie'))

    plt.scatter(samples_n, fouten_mt, label='montecarlo', c='b', marker='+')
    plt.scatter(samples_n, fouten_strat, label='montecarlo', c='tab:orange', marker='x')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N'), plt.ylabel('Fout')
    plt.title('De fout op de integratie van 0 tot {}'.format(b))
    plt.legend()
    plt.savefig('./plots/montecarlo/integratie_b={}.pdf'.format(b))
    plt.clf()


"""-----Toevalsgetallen-----"""


def rho(x):
    return 0.25 * np.pi * x * np.cos(0.125 * np.pi * x**2)


def rho_herschaald(x):
    return 0.125 * np.cos(0.125 * np.pi * x**2)


def rho_inv_cum(x):
    return 2 * f_inv_cum(x)


def toevalsgetallen(hist_type):
    u_samples = np.random.uniform(low=0.0, high=1.0, size=N)
    theta_samples = np.random.uniform(low=0.0, high=2 * np.pi, size=N)
    r_samples = rho_inv_cum(u_samples)

    x_samples_2d = np.multiply(r_samples, np.cos(theta_samples))
    y_samples_2d = np.multiply(r_samples, np.sin(theta_samples))

    if hist_type == 'doorsnede':
        hist_2d = np.histogram2d(x_samples_2d, y_samples_2d, bins=99, density=True)
        return hist_2d, x_samples_2d
    elif hist_type == 'colorplot':
        return x_samples_2d, y_samples_2d


def plot_2d_hist():
    x_samples, y_samples = toevalsgetallen(hist_type='colorplot')
    plt.hist2d(x_samples, y_samples, bins=99, cmap='viridis')
    plt.title('2D histogram plot van x en y')
    plt.xlabel('x'), plt.ylabel('y'),
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.savefig('./plots/toevalsgetallen/2d_hist_N={}.pdf'.format(N))
    plt.clf()


def plot_2d_doorsnede():
    hist, r_samples = toevalsgetallen(hist_type='doorsnede')
    bins_x, x_edges = hist[0][54], hist[1][:-1] / 2 + hist[1][1:] / 2
    r_samples.sort()
    rho_samples = rho_herschaald(r_samples)
    plt.xlabel('x'), plt.ylabel('waarde bin'), plt.title('2D doorsnede')
    plt.plot(x_edges, bins_x, marker='.', linestyle=':')
    plt.plot(r_samples, rho_samples)
    plt.savefig('./plots/toevalsgetallen/doorsnede_N={}.pdf'.format(N))
    plt.clf()


if __name__ == '__main__':
    main()
