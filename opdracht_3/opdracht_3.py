import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import gamma, beta
from scipy.special import gamma, beta
import time


SIZE = 100000
u_samples = np.random.uniform(low=0.0, high=1.0, size=SIZE)  # u samples via uniforme verdeling

"-----Config-----"
HM_PUNTEN_PLOT = True
HM_HIST = False
HM_INV_CUM = False


def main():
    if HM_PUNTEN_PLOT:
        plot_hit_or_miss_punten('uniform')
        plot_hit_or_miss_punten('triangulair')

    if HM_HIST:
        plot_hit_or_miss_hist('uniform')
        plot_hit_or_miss_hist('triangulair')

    if HM_INV_CUM:
        inv_cum()


def timer(func):
    """
    Een decorator om de performantie van een functie te testen. Deze functie neemt een andere functie als argument,
    voert de functie uit en print de tijd dat het duurde om deze uit te voeren.

    :param func: De functie waarvan we de performantie willen testen.
    """

    def wrapper(*args, **kwargs):
        print(args)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        param = ''
        if args:
            for arg in args:
                param += arg + ', '
        else:
            param = None

        print(('Het duurde {}s om de functie "{}" uit te voeren. '
               'De functie had de argumenten: "{}"').format(end_time - start_time, func.__name__, param[:-2]))
        return result
    return wrapper


def f(x):
    return np.pi * x * np.cos((np.pi / 2) * x ** 2)


def f_inv_cum(x):
    return np.sqrt((2 / np.pi) * np.arcsin(x))


def hx_triangular(x_samples):
    hx = np.array([3.2 * x if x <= 0.75 else -9.6 * x + 9.6 for x in x_samples])
    return hx


@timer
def hit_or_miss(verdeling):
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
        if y_samples[n] <= y_functie[n]:  # kijk of het punt een "hit" of een "miss" is
            x_hit.append(x_samples[n]), y_hit.append(y_samples[n])  # steek de hits in een lijst voor hit
        else:
            x_miss.append(x_samples[n]), y_miss.append(y_samples[n])  # steek de miss getallen in en lijst voor missers

    # efficientie
    print('efficientie = ', len(x_hit) / SIZE)
    return x_hit, y_hit, x_miss, y_miss


def plot_hit_or_miss_punten(verdeling):
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


def plot_hit_or_miss_hist(verdeling):
    x_hit, y_hit, x_miss, y_miss = hit_or_miss(verdeling)
    plt.hist(x_hit, bins=50, density=True, color='tab:orange')
    x = np.arange(0.0, 1.0, 0.001)
    plt.plot(x, f(x), linewidth=2.5, c='b')
    plt.title('Hit or Miss {} histogram'.format(verdeling, ))
    plt.xlabel('x'), plt.ylabel('f(x)')
    plt.grid()
    plt.legend(handles=[mpatches.Patch(color='orange', label='Histogram'),
                        mpatches.Patch(color='blue', label='f(x)')], loc='upper right')
    plt.show()


@timer
def inv_cum():
    x_samples = f_inv_cum(u_samples)
    plt.title('Inverse Cummulatieve'), plt.ylabel('f(x)'), plt.xlabel('x')
    plt.legend(handles=[mpatches.Patch(color='orange', label='Histogram'),
                        mpatches.Patch(color='blue', label='f(x)')], loc='upper right')
    plt.hist(x_samples, bins=50, density=True, color='tab:orange')

    t = np.arange(0.0, 1.0, 0.001)
    plt.plot(t, f(t), linewidth=2.5, c='b')
    plt.show()


if __name__ == '__main__':
    main()
