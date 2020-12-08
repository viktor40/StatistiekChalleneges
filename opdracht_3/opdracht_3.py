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
    4. timer decorator
    5. hit and miss methode
    6. monte carlo
    7. toevalsgetallen
    8. run het script
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np
import time

"-----Constanten-----"
# Stel het aantal punten in dat wordt gebruikt voor het trekken van random samples uit een verdeling bij hit and miss
SIZE = 50000


"-----Config-----"
rcParams.update({'font.size': 11})

# 1.1
HM_PUNTEN_PLOT = False
HM_HIST = False
HM_INV_CUM = False

# 1.2
MONTE_CARLO = True
STRAT = True
INT_FOUT = False

# 1.3
TOEVALSGETALLEN = False


def main():
    """
    In de main functie runnen we het script. Er kan in de config worden gekozen welke delen van het script moeten
    worden gerund.
    Voor de toevalsgetallen worden de functies uitgevoerd van 10E4 tot 10E8 zodat er een mooie evolutie te zien is.
    """
    # 1.1 genereren van random samples
    np.random.uniform(low=0.0, high=1.0, size=SIZE)

    # 1.1.1 Hit and miss
    if HM_PUNTEN_PLOT:
        plot_hit_or_miss_punten('uniform')
        plot_hit_or_miss_punten('triangulair')

    # 1.1.2 inverse cumulatieve + histogram
    if HM_INV_CUM:
        inv_cum()

    # 1.1.3 Hit and miss histogram
    if HM_HIST:
        plot_hit_or_miss_hist('uniform')
        plot_hit_or_miss_hist('triangulair')

    # 1.2.1 monte carlo methodes
    if MONTE_CARLO:
        u1 = monte_carlo(1, 100)
        u2 = monte_carlo(2, 100)
        print(u1, u2)

    if STRAT:
        u1 = stratificatie(1, 100)
        u2 = stratificatie(2, 100)
        print(u1, u2)

    # 1.2.2 Plotten van de foute op de monte carlo methodes
    if INT_FOUT:
        plot_integratie_fout(1, step=2)
        plot_integratie_fout(2, step=2)

    # 1.3 2D toevalsgetallen
    if TOEVALSGETALLEN:
        n = int(10E4)
        while n <= 10E8:
            plot_2d_hist(n)
            plot_2d_doorsnede(n)
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


"""-----Hit and Miss-----"""


def f(x):
    """
    De functie uit de opgave
    """
    return np.pi * x * np.cos((np.pi / 2) * x ** 2)


def f_inv_cum(x):
    """
    De inverse cummulatieve van de functie uit de opgave
    """
    return np.sqrt((2 / np.pi) * np.arcsin(x))


def hx_triangular(x_samples):
    """
    Een functie om hx te helpen berekenen voor een triangulaire verdeling.
    De top van de driehoek is bij x = 0.75.
    Als x kleiner is dan dit, dan kan de linker zijde van de driehoek benaderd worden door de functie f(x) = 3.2 * x.
    Als x groter is dan 0.75 wordt de rechter zijde benaderd door: f(x) = -9.6 * x + 9.6.
    Dit wordt op de dataset uitgevoerd zonder for-loops door np.where te gebruiken.
    :return: Een array met hierin de waarde van de driehoek die de functie benaderd, voor een bepaalde x-waarde.
    """
    return np.where(x_samples <= 0.75, 3.2 * x_samples, -9.6 * x_samples + 9.6)


@timer
def hit_or_miss(verdeling):
    """
    Deze functie past de hit and miss methode toe. Eerst worden er samples getrokken afhankelijk van de verdeling.
    Erna wordt er gekeken of het punt een hit of een miss is. Ten slotte wordt de efficientie nog berekend
    :param verdeling: Het type verdeling die we gebruiken voor de hit and miss. Dit is uniform of triangulair.
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

    # Door gebruik te maken van 'Fancy indexing' kunnen we het gebruik van for loops vermijden wanneer we bepalen of
    # een bepaald punt een hit of een miss is.
    x_hit, x_miss = x_samples[y_samples <= y_functie], x_samples[y_samples > y_functie]
    y_hit, y_miss = y_samples[y_samples <= y_functie], y_samples[y_samples > y_functie]

    # efficientie
    print('efficientie = ', len(x_hit) / SIZE)
    return x_hit, y_hit, x_miss, y_miss


def plot_hit_or_miss_punten(verdeling):
    """
    Deze functie plot een voorstelling van de hit and miss methode, waarbij de hits en misses zijn geplot alsook de
    functie.
    :param verdeling: Het type verdeling die we gebruiken voor de hit and miss. Dit is uniform of triangulair.
    """
    x_hit, y_hit, x_miss, y_miss = hit_or_miss(verdeling)
    plt.plot(x_hit, y_hit, ',', label='x', c='g')
    plt.plot(x_miss, y_miss, ',', label='x', c='r')
    x = np.arange(0.0, 1.0, 0.001)
    plt.plot(x, f(x), linewidth=2.5, c='b')
    plt.title('Hit and Miss {}'.format(verdeling))
    plt.xlabel('x'), plt.ylabel('f(x)')
    plt.legend(handles=[mpatches.Patch(color='green', label='Hit'),
                        mpatches.Patch(color='red', label='Miss'),
                        mpatches.Patch(color='blue', label='f(x)')], loc='upper right')
    plt.savefig('./plots/hit_or_miss/punten_{}.pdf'.format(verdeling))
    plt.clf()


def plot_hit_or_miss_hist(verdeling):
    """
    Deze functie plot een histogram van de hit and miss methode
    :param verdeling: Het type verdeling die we gebruiken voor de hit and miss. Dit is uniform of triangulair.
    """
    x_hit, y_hit, x_miss, y_miss = hit_or_miss(verdeling)
    plt.hist(x_hit, bins=50, density=True, color='tab:orange')
    x = np.arange(0.0, 1.0, 0.001)
    plt.plot(x, f(x), linewidth=2.5, c='b')
    plt.title('Hit and Miss {} histogram'.format(verdeling, ))
    plt.xlabel('x'), plt.ylabel('f(x)')
    plt.legend(handles=[mpatches.Patch(color='orange', label='Histogram'),
                        mpatches.Patch(color='blue', label='f(x)')], loc='upper right')
    plt.savefig('./plots/hit_or_miss/histogram_{}.pdf'.format(verdeling))
    plt.clf()


@timer
def inv_cum():
    """
    Deze functie berekent en plot de histogram voor de inverse cummulatieve
    """
    u_samples = np.random.uniform(low=0.0, high=1.0, size=SIZE)
    x_samples = f_inv_cum(u_samples)
    plt.title('Inverse Cumulatieve'), plt.ylabel('f(x)'), plt.xlabel('x')
    plt.legend(handles=[mpatches.Patch(color='orange', label='Histogram'),
                        mpatches.Patch(color='blue', label='f(x)')], loc='upper right')
    plt.hist(x_samples, bins=50, density=True, color='tab:orange')

    t = np.arange(0.0, 1.0, 0.001)
    plt.plot(t, f(t), linewidth=2.5, c='b')
    plt.savefig('./plots/hit_or_miss/inv_cum.pdf')
    plt.clf()


"""-----Monte carlo-----"""


def monte_carlo(b, n):
    """
    Deze functie voert de standaard monte carlo methode uit.
    :param b: De bovengrens van het integraal, 1 of 2. De ondergrens is altijd 0.
    :param n: Het aantal intervallen die we gebruiken.
    :return: De numerieke benadering van het integraal voor bepaalde b en n
    """
    xrand = np.random.uniform(0, b, (1, n))
    integral = np.sum(f(xrand))
    return b/n * integral


def stratificatie(b, n):
    """
    Deze functie voert de stratificatie methode uit. Het volledige gebied is niet in 2 opgedeeld maar in even en
    oneven.
    :param b: De bovengrens van het integraal, 1 of 2. De ondergrens is altijd 0.
    :param n: Het aantal intervallen die we gebruiken.
    :return: De numerieke benadering van het integraal voor bepaalde b en n
    """
    x_low = np.random.uniform(0, b/2, (1, int(np.floor(n/2))))
    x_high = np.random.uniform(b/2, b, (1, int(np.floor(n/2))))
    xrand = np.concatenate((x_low, x_high), axis=1)
    integral = np.sum(f(xrand))
    return b/n * integral


def fout_integratie(b, n, methode):
    """
    Deze functie zal de fout berekenen op de numerieke integratie voor een bepaalde n. Hiervoor wordt de integratie
    voor deze n 100 keer uitgevoerd en wordt de standaarddeviatie berekend.

    :param b: De bovengrens van het integraal, 1 of 2. De ondergrens is altijd 0.
    :param n: Het aantal intervallen die we gebruiken.
    :param methode: De integratiemethode, monte carlo of stratificatie
    :return: de fout als float.
    """
    if methode == 'monte carlo':
        data = np.array([monte_carlo(b, n) for _ in range(100)])
    elif methode == 'stratificatie':
        data = np.array([stratificatie(b, n) for _ in range(100)])
    else:
        raise Exception('Verkeerde Integratiemethode')
    return float(np.std(data))


@timer
def plot_integratie_fout(b, step):
    """
    De monte carlo integratie wordt een aantal keer uitgevoerd voor logaritmische stappen tussen 4 en 40 00.
    Telkens wordt de fout op de standaard monte carlo en de stratificatie geplot in functie van n.

    :param b: De bovengrens van het integraal, 1 of 2. De ondergrens is altijd 0.
    :param step: De logaritmische stappen. Het starpunt 4 wordt hiermee vermenigvuldigd.
    """
    print(f'----- fouten op integratie voor b={b} -----')
    i = 2
    samples_n = np.array([])
    fouten_mt, fouten_strat = np.array([]), np.array([])
    while i < 40000:
        print('Vooruitgang: i =', i, 'met logaritmische stappen van:', step)
        i *= step
        samples_n = np.append(samples_n, i)
        fouten_mt = np.append(fouten_mt, fout_integratie(b, i, 'monte carlo'))
        fouten_strat = np.append(fouten_strat, fout_integratie(b, i, 'stratificatie'))

    # converteer de data naar log data
    log_samples_n = np.log10(samples_n)
    log_fouten_mt = np.log10(fouten_mt)
    log_fouten_strat = np.log10(fouten_strat)

    # bereken de lineaire fit
    coeff_mt = np.polyfit(log_samples_n, log_fouten_mt, 1)
    coeff_strat = np.polyfit(log_samples_n, log_fouten_strat, 1)
    polynomial_mt = np.poly1d(coeff_mt)
    polynomial_strat = np.poly1d(coeff_strat)
    ys_mt = np.power(10, polynomial_mt(log_samples_n))
    ys_strat = np.power(10, polynomial_strat(log_samples_n))

    print('lineaire fit monte carlo: f(x) =', polynomial_mt)
    print('lineaire fit stratificatie: f(x) =', polynomial_strat)

    # plot de gefitte rechtes
    plt.plot(samples_n, ys_mt, label=str(polynomial_mt), ls=':', c='tab:blue')
    plt.plot(samples_n, ys_strat, label=str(polynomial_strat), ls=':', c='tab:orange')

    # plot de punten
    plt.scatter(samples_n, fouten_mt, label='monte carlo', c='b', marker='+')
    plt.scatter(samples_n, fouten_strat, label='stratificatie', c='orangered', marker='x')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N'), plt.ylabel('Fout')
    plt.title('De fout op de integratie van 0 tot {}'.format(b))
    plt.legend()
    plt.savefig('./plots/monte_carlo/integratie_b={}.pdf'.format(b))
    plt.clf()


"""-----Toevalsgetallen-----"""


def rho(x):
    """
    De functie voor rho.
    """
    return 0.25 * np.pi * x * np.cos(0.125 * np.pi * x**2)


def rho_beter(x):
    """
    De veranderde functie voor rho die de 2D doorsnede wel benaderd.
    """
    return 0.125 * np.cos(0.125 * np.pi * x**2)


def rho_inv_cum(x):
    """
    De inverse cummulatieve van rho.
    """
    return 2 * f_inv_cum(x)


def toevalsgetallen(n):
    """
    Deze funcite berekend de 2D toevalsgetallen. Samples worden getrokken voor r en theta en dan omgerekend naar x en y.
    :return: de samples voor x en y omgerekend uit r en theta
    """
    u_samples = np.random.uniform(low=0.0, high=1.0, size=n)
    theta_samples = np.random.uniform(low=0.0, high=2*np.pi, size=n)
    r_samples = rho_inv_cum(u_samples)
    x_samples_2d = np.multiply(r_samples, np.cos(theta_samples))
    y_samples_2d = np.multiply(r_samples, np.sin(theta_samples))
    return x_samples_2d, y_samples_2d


def plot_2d_hist(n):
    """
    Deze functie zal het 2D histogram plotten door gebruik te maken van kleuren.
    """
    x_samples, y_samples = toevalsgetallen(n)
    plt.hist2d(x_samples, y_samples, bins=99, cmap='viridis')
    plt.title('2D histogram plot van x en y')
    plt.xlabel('x'), plt.ylabel('y'),
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.savefig('./plots/toevalsgetallen/2d_hist_N={}.pdf'.format(n))
    plt.clf()


def plot_2d_doorsnede(n):
    """
    Deze functie zal de doorsnede van het 2D histogram maken en plotten. Eerst wordt het 2D histogram gevormd via
    numpy. Vervolgens trekken we er een doorsnede van bins uit (we trekken ze uit het midden). Ook worden de
    edges bepaald. We gebruiken een simpele uitmiddeling door de edges en de omgekeerde lijst op te tellen en te delen
    door 2.
    Ten slotte wordt de theoretische functie en de benadering geplot.
    """
    r_samples, y_samples = toevalsgetallen(n)
    hist = np.histogram2d(r_samples, y_samples, bins=99, density=True)
    bins_x, x_edges = hist[0][54], hist[1][:-1] / 2 + hist[1][1:] / 2
    r_samples.sort()
    rho_samples = rho_beter(r_samples)
    plt.xlabel('x'), plt.ylabel('waarde bin'), plt.title('2D doorsnede')
    plt.plot(x_edges, bins_x, marker='.', linestyle=':')
    plt.plot(r_samples, rho_samples)
    plt.savefig('./plots/toevalsgetallen/doorsnede_N={}.pdf'.format(n))
    plt.clf()


if __name__ == '__main__':
    main()
