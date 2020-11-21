"""
01/12/2020

Python Challenges Statistiek en Gegevensverwerking
Opdracht 2
Groep 4: Viktor Van Nieuwenhuize
         Aiko Decaluwe
         Fien Dewit

Indeling .py bestand:
    1. constanten
    2. config
    3. main function
    4. bepalen van de schatters
    5. bepalen van de covariantiematrix van de schattingsmethode
    6. bootstrappen
    7. vergelijken van de geschatte waarden t.o.v. de gebootstrapte waarden
    8. run het script
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import digamma, polygamma
import scipy.optimize as optimize
import pickle


"""----- Constanten -----"""
samples = np.loadtxt('./gamma_samples.dat')
M = 100
N_START = 100
N_MAX = 100000
STEPS = 500


"""----- Config -----"""
momenten = True
likelihood = True

bootstrappen = False
plotten = False
vergelijken = True


def main():
    """
    Hier wordt het script uitgevoerd. Datapunten worden opgeslaan via pickle. Als boostrappen False is worden deze
    niet opnieuw berekend.
    Verder kan men in config kiezen om niet te plotten of de gebootstrapte varianties niet te vergelijken met de
    geschatte varianties.
    """

    # Methode van de Momenten
    if momenten:
        if bootstrappen:
            bootstrap_result = repeat_bootstrap(schatters_MM)
            outfile = open('bootstrap_MM', 'wb')
            pickle.dump(bootstrap_result, outfile)
            outfile.close()
        else:
            inputfile = open('bootstrap_MM', 'rb')
            bootstrap_result = pickle.load(inputfile)
            inputfile.close()

        if plotten:
            plot(bootstrap_result, 'MM')

        if vergelijken:
            vergelijk('MM')

    # divider
    if likelihood and momenten and vergelijken:
        print('-' * 100)

    # Maximum Likelihood Methode
    if likelihood:
        if bootstrappen:
            bootstrap_result = repeat_bootstrap(schatters_MLLH)
            outfile = open('bootstrap_MLLH', 'wb')
            pickle.dump(bootstrap_result, outfile)
            outfile.close()
        else:
            inputfile = open('bootstrap_MLLH', 'rb')
            bootstrap_result = pickle.load(inputfile)
            inputfile.close()

        if plotten:
            plot(bootstrap_result, 'MLLH')

        if vergelijken:
            vergelijk('MLLH')


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
    """
    Deze functie trekt N random samples uit de gegeven dataset.
    :param N: Het aantal samples dat we willen trekken uit de dataset
    """
    return np.random.choice(samples, size=N)


def schatters_MLLH(data):
    """
    Deze functie berekend de schatters voor de maximum likelihood methode. k moet numeriek berekend worden.
    Dit wordt gedaan in de functie schatter_k_MLLH.
    Er werd gekozen voor scipy.optimize.broyden2 na het testen van verschillende methodes en het vergelijken van de
    performantie. broyden2 kwam er uit als de snelste voor deze vergelijking.

    :param data: de dataset waaruit k en theta wordt berekend
    :return: De schatters k en theta
    """
    y = data
    gem_y = np.average(y)

    def schatter_k_MLLH(k):
        """
        :param k: de schatter k
        :return: de vergelijking die numeriek moet opgelost worden
        """
        gem_lny = np.average(np.log(y))
        f = gem_lny - np.log(gem_y) + np.log(k) - digamma(k)
        return f

    k = float(optimize.broyden2(schatter_k_MLLH, np.array([3.0])))
    theta = gem_y / k
    return k, theta


def covariantie_momenten_MM(xi, xi_2, gem_x, gem_x2, N):
    """
    Deze functie berekent de covariantie van de momenten gem(x) en gem(x^2) voor gebruik in de formule
    om de covariantie van MM schatters te berekenen (vergelijking 6.5)

    :param xi: aan array met hiering alle waarden uit de dataset
    :param xi_2: een array met hierin de waarden van de dataset gekwadrateerd
    :param gem_x: het gemiddelde van de array xi
    :param gem_x2: het gemiddelde van de array xi_2
    :param N: het aantal waarden in de dataset
    :return: variantie k, covariantie, variantie theta
    """
    V_11 = 1 / (N * (N - 1)) * np.sum((xi - gem_x)**2)  # cov(k, k)
    V_12 = 1 / (N * (N - 1)) * np.sum((xi - gem_x) * (xi_2 - gem_x2))  # cov(k, theta)
    V_22 = 1 / (N * (N - 1)) * np.sum((xi_2 - gem_x2) ** 2)  # cov(theta, theta)
    return V_11, V_12, V_22


def afgeleides_MM_schatters(gem_x, gem_x2):
    """
    De formules voor de afgeleides van de schatters k en theta naar de momenten gem(x) en gem(x^2) zijn hier
    geïmplementeerd. Deze worden gebruikt voor de covariantie te berekenen (formule 6.5)

    :param gem_x: Het gemiddelde van alle x-waarden
    :param gem_x2: Het gemiddelde van de kwadraten van de x-waarden
    """
    dk_dx = (2 * gem_x * gem_x2) / ((gem_x2 - gem_x ** 2) ** 2)
    dk_dx2 = - gem_x ** 2 / ((gem_x2 - gem_x ** 2) ** 2)
    dt_dx = - (gem_x2 + gem_x ** 2) / gem_x ** 2
    dt_dx2 = 1 / gem_x
    return dk_dx, dk_dx2, dt_dx, dt_dx2


def covariantie_MM():
    """
    Deze functie implementeerd formule 6.5 uit de cursus om de covariantie van de MM schatters k en theta te berekenen.
    """
    xi = samples
    xi_2 = samples**2
    N = len(samples)

    gem_x = np.average(xi)
    gem_x2 = np.average(xi_2)

    dk_dx, dk_dx2, dt_dx, dt_dx2 = afgeleides_MM_schatters(gem_x, gem_x2)
    cov_x_x, cov_x_x2, cov_x2_x2 = covariantie_momenten_MM(xi, xi_2, gem_x, gem_x2, N)

    var_k = (dk_dx ** 2)*cov_x_x + (dk_dx2 ** 2)*cov_x2_x2 + (2 * dk_dx * dk_dx2)*cov_x_x2
    var_theta = (dt_dx ** 2)*cov_x_x + (dt_dx2 ** 2)*cov_x2_x2 + (2 * dt_dx * dt_dx2)*cov_x_x2
    cov_k_theta = (dk_dx * dt_dx)*cov_x_x + (dk_dx2 * dt_dx2)*cov_x2_x2 + (dk_dx * dt_dx2 + dk_dx2 * dt_dx) * cov_x_x2
    cor_k_theta = cov_k_theta / np.sqrt(var_k * var_theta)

    return var_k, var_theta, cor_k_theta


def covariantie_MLLH():
    """
    Deze formule implementeerd de vergelijkingen om de inverse covariantiematrix uit te rekenen voor MLLH schatters.
    (Stelling 6.2.5)
    Hierna wordt deze matrix geïnverteerd waarna we de variantie van de schatters uit deze matrix kunnen halen.
    :return: de variantie van k en theta alsook de covariantie tussen de 2
    """

    k, theta = schatters_MLLH(samples)
    N = len(samples)

    # de elementen van V^-1
    C_11 = N * polygamma(1, k)
    C_12 = N / k
    C_22 = N * (k / theta ** 2)

    # inverse covariantiematrix
    cov_inv = np.array([[C_11, C_12],
                        [C_12, C_22]])

    # inverteer de inverse covariantiematrix
    cov = np.linalg.inv(cov_inv)

    # haal de covariantie en variantie uit de matrix en bereken de correlatie
    var_k = cov[0][0]
    var_theta = cov[1][1]
    cov_k_theta = cov[0][1]
    cor_k_theta = cov_k_theta / np.sqrt(var_k * var_theta)
    return var_k, var_theta, cor_k_theta


def schatters_MM(data):
    """
    Deze functie berekent de schatters voor de methode van de momenten
    :param data: de dataset waaruit k en theta wordt berekend
    :return: De schatters k en theta
    """
    gem_x = np.average(data)
    gem_x2 = np.average(data**2)

    k = gem_x**2 / (gem_x2 - gem_x ** 2)
    theta = (gem_x2 - gem_x ** 2) / gem_x
    return k, theta


def bias(k_bootstrap, theta_bootstrap, echte_k, echte_theta):
    """
    Deze functie berekend de bias van k en theta

    :param k_bootstrap: de waarde van k die werd bekomen door het bootstrappen
    :param theta_bootstrap: De waarde van theta die werd bekomen door het bootstrappen
    :param echte_k: de echte waarde van k, de waarde die men uitkomt voor alle waarden van k
    :param echte_theta: de echte waarde van theta, de waarde die men uitkomt voor alle waarden van theta
    :return: de bias van k en theta
    """
    bias_k = np.average(k_bootstrap) - echte_k
    bias_theta = np.average(theta_bootstrap) - echte_theta
    return bias_k, bias_theta


def bootstrap(N, func, echte_k, echte_theta):
    """
    Deze functie past het bootstrappen toe. k en theta worden M keer berekend en er worden N waarden getrokken
    uit de samples. Hieruit wordt de bias en variantie van k en theta berekend alsook de correlatie tussen k en theta.

    :param N: Het aantal random waarden uit de samples die worden gebruikt voor de bootstrap
    :param func: de functie die wordt gebruikt om de schatters van k en theta te berekenen
    :param echte_k: de echte waarde van k, de waarde die men uitkomt voor alle waarden van k
    :param echte_theta: de echte waarde van theta, de waarde die men uitkomt voor alle waarden van theta
    :return: de bias van k en theta, de variantie van k en theta en de correlatie tussen k en theta
    """
    k_bootstrap, theta_bootstrap = np.array([func(pick_random_samples(N)) for _ in range(M)]).T
    bias_k, bias_theta = bias(k_bootstrap, theta_bootstrap, echte_k, echte_theta)
    var_k = np.var(k_bootstrap)
    var_theta = np.var(theta_bootstrap)
    corr = np.corrcoef([k_bootstrap, theta_bootstrap])
    return bias_k, bias_theta, var_k, var_theta, corr[0][1]


@timer
def repeat_bootstrap(func):
    """
    Deze functie zal N keer de bootstrap toepassen voor verschillende aantallen N. De verscillende N die worden gebruikt
    zijn bepaald via een step. Deze functie wordt ook getimed.

    :param func: de functie die wordt gebruikt om de schatters van k en theta te berekenen
    :return: een array met lijsten voor de biassen, varianties en correlaties van k en theta.
    """
    echte_k, echte_theta = func(samples)
    repeat_data = np.array([bootstrap(N, func, echte_k, echte_theta) for N in range(N_START, N_MAX, STEPS)]).T
    return repeat_data


def plot(data, methode):
    """
    Deze functie plot de bias van k en theta, de variantie van k en theta en de correlatie tussen de 2.

    :param data: De waarden om te plotten. Data bevat lijsten voor de biassen, de varianties en de correlaties.
    :param methode: De methode die gebruikt werd om de data te berekenen. Dit wordt gebruikt voor titels en het opslaan
                    van de plots.
    """
    bias_k, bias_theta, var_k, var_theta, corr = data

    x_values = np.array(range(N_START, N_MAX, STEPS))

    # Bias plotten
    plt.scatter(x_values, bias_k, c='red', marker='.', label='k')
    plt.scatter(x_values, bias_theta, c='blue', marker='.', label='theta')
    plt.xlabel('N iteraties'), plt.ylabel('Bias')
    plt.title('Bias' + methode)
    plt.legend(loc='upper right')
    plt.savefig("plots/Bias_{}.pdf".format(methode), bbox_inches="tight")
    plt.clf()

    # Variantie plotten
    plt.scatter(x_values, var_k, c='red', marker='.', label='k')
    plt.scatter(x_values, var_theta, c='blue', marker='.', label='theta')
    plt.xlabel('N iteraties'), plt.ylabel('Variantie')
    plt.title('Variantie ' + methode)
    plt.legend(loc='upper right')
    plt.savefig("plots/Variantie_{}.pdf".format(methode), bbox_inches="tight")
    plt.clf()

    # Covariantie plotten
    plt.scatter(x_values, corr, c='red', marker='.')
    plt.xlabel('N iteraties'), plt.ylabel('Correlatiecoëfficiënt')
    plt.title('Correlatiecoëfficiënt ' + methode)
    plt.savefig("plots/Correlatie_{}.pdf".format(methode), bbox_inches="tight")
    plt.clf()


def vergelijk(schattingsmethods):
    """
    In deze functie worden de variantie en covariantie van k en theta vergeleken. Er wordt een vergelijking gemaakt
    tussen de geschatte waarde en de gebootstrapte waarde.
    :param schattingsmethods: De methode die gebruikt moet worden om de variantie en covariantie te bepalen.
    """

    # bepalen welke functie moet gebruikt worden om k en theta te bepalen alsook zijn variantie en covariantie
    if schattingsmethods == 'MM':
        schatters = schatters_MM
        variantie = covariantie_MM

    elif schattingsmethods == 'MLLH':
        schatters = schatters_MLLH
        variantie = covariantie_MLLH

    else:
        raise Exception('Verkeerde schattingsmethode')

    # de ware waarde van k en theta
    echte_k, echte_theta = schatters(samples)

    # de gebootstrapte varianties
    bootstrap_var = bootstrap(N_MAX, schatters, echte_k, echte_theta)
    var_k_bootstrap = bootstrap_var[2]
    var_theta_bootstrap = bootstrap_var[3]
    corr_bootstrap = bootstrap_var[4]

    # de geschatte varianties
    var_k_schatters, var_theta_schatters, corr_schatters = variantie()

    print('Variantie k      -  bootstrap: {}            schatting: {}'.format(var_k_bootstrap, var_k_schatters))
    print('Variantie theta  -  bootstrap: {}            schatting: {}'.format(var_theta_bootstrap, var_theta_schatters))
    print('correlatie       -  bootstrap: {}            schatting: {}'.format(corr_bootstrap, corr_schatters))


if __name__ == '__main__':
    main()
