"""
17/11/2020

Python Challenges Statistiek en Gegevensverwerking
Opdracht 1
Groep 4: Viktor Van Nieuwenhuize
         Aiko Decaluwe
         Fien Dewit
         Isabelle Vanderhaeghen

Indeling .py bestand:
    1. imports
    2. constanten
    3. functies voor berekeningen
    4. functies voor plotten
    5. Uitvoeren van juiste functies per deelopdracht

Indien een parameter van een functie geen np.array([]) is, wordt deze meestal gebruikt om aan te duiden voor wat de
functie wordt gebruikt. Voor duidelijkheid weren in dat geval type hints gebruikt bij de parameters.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle


# ---- Constanten ----
# de fouten op r, theta en sigma
SIGMA_R, SIGMA_THETA, SIGMA_PHI = 0.001, 0.003, 0.005

# de covariantiematrix voor de sferische coordinaten
CR = np.array([[SIGMA_R ** 2, 0, 0],
               [0, SIGMA_THETA ** 2, 0],
               [0, 0, SIGMA_PHI ** 2]])

# de covariantiematrix voor de sferische coordinaten met systematische fouten
CR_S = np.array([[SIGMA_R ** 2, 0, 0],
                 [0, SIGMA_THETA ** 2 + 0.01 ** 2, 0.01 ** 2],
                 [0, 0.01 ** 2, SIGMA_PHI ** 2 + 0.01 ** 2]])


# Laadt de dataset als een numpy array. Hierna wordt de data getransponeerd. Dit wordt gedaan zodat alle
# r, theta en phi waarden dan in dezelfde subarray zitten, i.e. r = data[0], theta = data[1] en phi = data[2].
punten_sferisch = np.loadtxt('./punten_sferisch.dat')
data = np.transpose(punten_sferisch)


def sferisch_to_cartesisch(r, theta, phi):
    """
    De coordinaattransformatie van sferische naar cartesische coordinaten wordt in deze functie geïmplementeerd.

    :param r: een 504 x 1 array met hierin alle waarden voor r
    :param theta: een 504 x 1 array met hierin alle waarden voor theta
    :param phi: een 504 x 1 array met hierin alle waarden voor phi
    :return: een 504 x 3 array. Elke 1 x 3 matrix bevat x, y en z, m.a.w 1 cartesische coordinaat.
    """

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def afgeleide_r(theta, phi):
    """
    x, y en z afleiden naar r.
    :param theta: een 504 x 1 array met hierin alle waarden voor theta
    :param phi: een 504 x 1 array met hierin alle waarden voor phi
    :return: een 504 x 3 array. Elke 1 x 3 matrix bevat de afgeleide van x, y en z naar r voor een coordinaat.
    """

    dx_dr = np.sin(theta) * np.cos(phi)
    dy_dr = np.sin(theta) * np.sin(phi)
    dz_dr = np.cos(theta)
    return np.array([dx_dr, dy_dr, dz_dr])


def afgeleide_theta(r, theta, phi):
    """
    x, y en z afleiden naar theta.
    :param r: een 504 x 1 array met hierin alle waarden voor r
    :param theta: een 504 x 1 array met hierin alle waarden voor theta
    :param phi: een 504 x 1 array met hierin alle waarden voor phi
    :return: een 504 x 3 array. Elke 1 x 3 matrix bevat de afgeleide van x, y en z naar theta voor een coordinaat.
    """

    dx_dt = r * np.cos(theta) * np.cos(phi)
    dy_dt = r * np.cos(theta) * np.sin(phi)
    dz_dt = - r * np.sin(theta)
    return np.array([dx_dt, dy_dt, dz_dt])


def afgeleide_phi(r, theta, phi):
    """
    x, y en z afleiden naar phi.
    :param r: een 504 x 1 array met hierin alle waarden voor r
    :param theta: een 504 x 1 array met hierin alle waarden voor theta
    :param phi: een 504 x 1 array met hierin alle waarden voor phi
    :return: een 504 x 3 array. Elke 1 x 3 matrix bevat de afgeleide van x, y en z naar phi voor een coordinaat.
    """

    dx_dp = - r * np.sin(theta) * np.sin(phi)
    dy_dp = r * np.sin(theta) * np.cos(phi)
    dz_dp = 0.0 * r
    return np.array([dx_dp, dy_dp, dz_dp])


def matrix_vermenigvuldiging(systematische_fout: bool):
    """
    Deze functie implementeert de formule voor de foutenpropagatie als matrix vermenigvuldiging.
    Eerst wordt een array A gecreëerd. Deze array A zal een 3 x 3 x 504 array zijn.

    Idien men A transponeerd wordt dit een 504 x 3 x 3 array waarin elke 3 x 3 matrix de vorm heeft van de
    volgende matrix:
    M = [[dx/dr, dx/dt, dx/dp]
         [dy/dr, dy/dt, dy/dp]
         [dz/dr, dz/dt, dz/dp]]

    We vermenigvuldigen deze matrix M met CR en dan met zijn getransponeerde M.T
    Dit gebeurd voor elke 3x3 matrix in de 504 x 3 x 3 array, zodat we weer een 504 x 3 x 3 array terug krijgen
    waarbij elke 3 x 3 matrix de covariantiematrix CX van die coördinaat voorstelt.

    Al deze bewerkingen worden via een einsum gedaan aangezien dit veel tijd bespaart. Al deze bewerkingen in één
    keer uitvoeren levert het voordeel op dat er slechts 1 keer over alle waarden moet worden geïtereerd.

    Ten slotte pickelen we de geresulteerde array. Hierdoor kan hij makkelijk weer opgeroepen worden om te gebruiken
    tijdens het plotten zonder dat hij opnieuw berekend moet worden.

    :param systematische_fout: Als deze parameter True is wordt de covariantiematrix met systematische fout voor de
           spherische coordinaten berekend.
    """

    # de sferische coordinaten als 1 x 504 array
    r = data[0]
    theta = data[1]
    phi = data[2]

    # stel de matrix op om te gebruiken in de matrixvermenigvuldiging
    A = np.array(([afgeleide_r(theta, phi),
                   afgeleide_theta(r, theta, phi),
                   afgeleide_phi(r, theta, phi)]))

    # pas de einsum toe om de uiteindelijke covariantiematrices te verkrijgen als 508 x 3 x 3 array.
    if not systematische_fout:
        # Wanneer er geen systematische fout is gebruiken we CR.
        CX = np.einsum('kji, kl, lmi -> ijm', A, CR, A)
        outfile = open("covariantiematrix_geen_correlaties", 'wb')
        pickle.dump(CX, outfile)
        outfile.close()

    else:
        # Wanneer er wel een systematische fout is gebruiken we CR_S
        CX = np.einsum('kji, kl, lmi -> ijm', A, CR_S, A)
        outfile = open("covariantiematrix_systematische_fout", 'wb')
        pickle.dump(CX, outfile)
        outfile.close()


def plot_coord_2d(x, y, assen: str, color: str):
    """
    Deze functie plot de 2D projecties van de cartesische coordinaten.
    :param x: De x waarden voor het plot
    :param y: De y waarden voor het plot
    :param assen: De assen die het vlak maken waarop de data wordt geprojecteerd
    :param color: Het kleur voor de punten op het scatterplot
    """

    plt.scatter(x, y, marker='.', color=color)
    plt.xlabel('{}-waarden'.format(assen[0])), plt.ylabel('{}-waarden'.format(assen[1]))
    plt.title('De spreiding van de {} en {} waarden.'.format(assen[0], assen[1]))
    plt.savefig('plots/deel1/{}_spreiding.pdf'.format(assen))
    plt.clf()


def coordinaattransformatie():
    """
    Deze functie bevat de code voor het uitvoeren van de coordinaattransformatie. Deze werd in een functie gezet
    zodat men deze enkel hoeft uit te voeren wanneer nodig en de grafiek niet gemaakt zal worden als de call verwijderd
    wordt of in een comment geplaatst wordt.

    Deze functie plot de spreiding van de punten als een histogram. Voor elke cartesische coordinaat is er een
    histogram. Deze histogrammen worden deels doorzichtig gemaakt en over elkaar gelegd om de spreidingen tussen de
    coordinaten gemakkelijk te kunnen bekijken. Het plot wordt als PDF opgeslaan zodat deze "scalable" is.

    Deze functie plot ook de XYZ putnen in 3D. Ten slotte worden plotjes gemaakt met de spreiding voor XY, YZ en ZX
    """

    # Histogrammen van X, Y en Z
    punten_cartesisch = sferisch_to_cartesisch(data[0], data[1], data[2])
    plt.hist(punten_cartesisch[0], bins=30, density=True, alpha=0.25, color='green', histtype='bar')
    plt.hist(punten_cartesisch[0], bins=30, density=True, alpha=1, color='green', histtype='step')
    plt.hist(punten_cartesisch[1], bins=30, density=True, alpha=0.25, color='red', histtype='bar')
    plt.hist(punten_cartesisch[1], bins=30, density=True, alpha=1, color='red', histtype='step')
    plt.hist(punten_cartesisch[2], bins=30, density=True, alpha=0.25, color='blue', histtype='bar')
    plt.hist(punten_cartesisch[2], bins=30, density=True, alpha=1, color='blue', histtype='step')
    plt.legend(handles=[mpatches.Patch(color='green', label='x'),
                        mpatches.Patch(color='red', label='y'),
                        mpatches.Patch(color='blue', label='z')],
               loc='upper right')

    plt.xlabel('Coordinaat'), plt.ylabel('Waarschijnlijkheidsdichtheid')
    plt.title('Histogrammen van de cartesische coördinaten')
    plt.savefig('plots/deel1/xyz_histogrammen.pdf')
    plt.clf()

    # XYZ scatterplot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(punten_cartesisch[0], punten_cartesisch[1], punten_cartesisch[2], depthshade=True, marker='x')
    ax.set_xlabel('X-waarden'), ax.set_ylabel('Y-waarden'), ax.set_zlabel('Z-waarden')
    ax.set_title('De spreiding van de cartesische coordinaten.')
    plt.savefig('plots/deel1/xyz_spreiding.pdf')
    plt.clf()

    # XY scatterplot
    plot_coord_2d(punten_cartesisch[0], punten_cartesisch[1], assen='xy', color='green')

    # YZ scatterplot
    plot_coord_2d(punten_cartesisch[1], punten_cartesisch[2], assen='yz', color='red')

    # ZX scatterplot
    plot_coord_2d(punten_cartesisch[2], punten_cartesisch[0], assen='zx', color='blue')


def plot_fouten(x_waarde, y_waardes, systematische_fout: bool, spherische_coord: str):
    """
    Deze functie kan gebruikt worden voor het plotten van de fouten op de cartesische coördinaten in functie van
    de sferische coördinaten.

    :param x_waarde: Een numpy array met de waarden voor de cartesische coordinaat
    :param y_waardes: Een tupel met 3 arrays van de waarden voor de cartesische coordinaten x, y en z respectievelijk.
    :param systematische_fout: Of de covariantiematrix van de sferische coordinaten die gebruikt werd al dan niet een
                               systematische fout had
    :param spherische_coord: In functie van welke sferische coordinaat de cartesische coordinaten geplot worden.
                             Deze variabele wordt gebruikt voor de labels en file names.
    """
    plt.scatter(x_waarde, y_waardes[0], marker='.', color='royalblue', label='x')
    plt.scatter(x_waarde, y_waardes[1], marker='.', color='tab:green', label='y')
    plt.scatter(x_waarde, y_waardes[2], marker='.', color='darkorange', label='z')
    plt.xlabel('{} coördinaten'.format(spherische_coord)), plt.ylabel('x, y en z coördinaten')
    plt.title('De fouten van de x, y en z coordinaten in functie van {}{}.'.format(spherische_coord, ' met systematische fout' if systematische_fout else ''))
    plt.legend(loc='upper right')
    plt.savefig('plots/{}/fout_ifv_{}.pdf'.format('met S fout' if systematische_fout else 'zonder S fout',
                                                  spherische_coord), bbox_inches="tight")
    plt.clf()


def plot_correlaties(x_waarde, y_waardes, systematische_fout: bool, spherische_coord: str):
    """
    Deze functie kan gebruikt worden voor het plotten van de correlaties tussen de cartesische coördinaten in functie
    van de sferische coördinaten.

    :param x_waarde: Een numpy array met de waarden voor de cartesische coordinaat
    :param y_waardes: Een tupel met 3 arrays van de waarden voor de cartesische coordinaten x, y en z respectievelijk.
    :param systematische_fout: Of de covariantiematrix van de sferische coordinaten die gebruikt werd al dan niet een
                               systematische fout had
    :param spherische_coord: In functie van welke sferische coordinaat de cartesische coordinaten geplot worden.
                             Deze variabele wordt gebruikt voor de labels en file names.
    """
    plt.scatter(x_waarde, y_waardes[0], marker='.', color='royalblue', label='xy')
    plt.scatter(x_waarde, y_waardes[1], marker='.', color='tab:green', label='yz')
    plt.scatter(x_waarde, y_waardes[2], marker='.', color='darkorange', label='zx')
    plt.xlabel('{} coördinaten'.format(spherische_coord))
    plt.ylabel('de correlaties tussen xy, yz en zx'.format(spherische_coord))
    plt.title('De correlaties in functie van {}{}.'.format(spherische_coord,
                                                           '\nmet systematische fout' if systematische_fout else ''))
    plt.legend(loc='upper right')
    plt.savefig('plots/{}/correlaties_ifv_{}.pdf'.format('met S fout' if systematische_fout else 'zonder S fout',
                                                         spherische_coord), bbox_inches="tight")
    plt.clf()


def plot_cov(matrices, systematische_fout: bool):
    """
    In deze functie wordt het plotten van de waarden in de covariantiematrix uitgevoerd, zowel met als zonder
    systematische fouten. Eerst wordt de 503 x 3 x 3 array getransponeerd tot 3 x 3 x 504 array zodat we door
    de array te indexeren makkelijk alle waarden voor een bepaalde covariantie krijgen. Het is dan eenvoudig om de
    1 x 504 arrays door te geven aan de plotfuncties om te gebruiken tijdens het plotten.

    Na het transponeren heeft cov_per_coordinaat de vorm:
    [[[Cov_XX_1, Cov_XX_2, ... , Cov_XX_508],
      [Cov_XY_1, Cov_XY_2, ... , Cov_XY_508],
      [Cov_XZ_1, Cov_XZ_2, ... , Cov_XZ_508]]

     [[Cov_YX_1, Cov_YX_2, ... , Cov_YX_508],
      [Cov_YY_1, Cov_YY_2, ... , Cov_YY_508],
      [Cov_YZ_1, Cov_YZ_2, ... , Cov_YZ_508]]

     [[Cov_ZX_1, Cov_ZX_2, ... , Cov_ZX_508],
      [Cov_ZY_1, Cov_ZY_2, ... , Cov_ZY_508],
      [Cov_ZZ_1, Cov_ZZ_2, ... , Cov_ZZ_508]]

    De fouten kunnen bepaald worden door de vierkantswortels te nemen van  Cov_XX, Cov_YY en Cov_ZZ.

    Voor de correlaties worden de niet diagonaalelementen gebruikt. Aangezien de covariantiematrix symmetrisch is
    maat het niet uit of we bijvoorbeeld Cov_XY of Cov_YX gebruiken. Deze covariantie wordt dan gedeeld door de
    vierkantswortel van het product van de varianties om de correlatie te bepalen.
    vb: Corr_XY = Cov_XY / np.sqrt(Cov_XX * Cov_YY)

    :param matrices: De 504 x 3 x 3 array die werd berekend in de matrix_vermenigvuldiging functie
    :param systematische_fout: Of de covariantiematrix van de sferische coordinaten die gebruikt werd al dan niet een
                               systematische fout had.
    """
    cov_per_coordinaat = np.transpose(matrices, (1, 2, 0))

    # haal de 1 x 504 arrays uit de array om te gebruiken bij het plotten
    var_x = cov_per_coordinaat[0][0]
    var_y = cov_per_coordinaat[1][1]
    var_z = cov_per_coordinaat[2][2]
    cov_xy = cov_per_coordinaat[0][1]
    cov_yz = cov_per_coordinaat[1][2]
    cov_zx = cov_per_coordinaat[2][0]

    # bereken de fouten
    fout_x = np.sqrt(var_x)
    fout_y = np.sqrt(var_y)
    fout_z = np.sqrt(var_z)

    # bereken de correlaties
    corr_xy = cov_xy / np.sqrt(var_x * var_y)
    corr_yz = cov_yz / np.sqrt(var_y * var_z)
    corr_zx = cov_zx / np.sqrt(var_z * var_x)

    # de waarden voor de sferische coordinaten
    r = data[0]
    theta = data[1]
    phi = data[2]

    # plots van de fouten in functie van de spherische coordinaten
    plot_fouten(r, (fout_x, fout_y, fout_z), systematische_fout, 'r')
    plot_fouten(theta, (fout_x, fout_y, fout_z), systematische_fout, 'theta')
    plot_fouten(phi, (fout_x, fout_y, fout_z), systematische_fout, 'phi')

    # plots van de correlaties in functie van de sferische coordinaten
    plot_correlaties(r, (corr_xy, corr_yz, corr_zx), systematische_fout, 'r')
    plot_correlaties(theta, (corr_xy, corr_yz, corr_zx), systematische_fout, 'theta')
    plot_correlaties(phi, (corr_xy, corr_yz, corr_zx), systematische_fout, 'phi')


# voor de coordinaattransformatie uit (Puntje 2.1)
coordinaattransformatie()


# bereken de covariantiematrices (Puntje 2.2 en 2.3)
matrix_vermenigvuldiging(systematische_fout=False)


# plot de covariantiematrices (Puntje 3)
inputfile = open("covariantiematrix_geen_correlaties", 'rb')
cov_matrices = pickle.load(inputfile)
inputfile.close()
plot_cov(cov_matrices, systematische_fout=False)


# plot de covariantiematrices met systematische fout (Puntje 4)
matrix_vermenigvuldiging(systematische_fout=True)
inputfile = open("covariantiematrix_systematische_fout", 'rb')
cov_matrices = pickle.load(inputfile)
inputfile.close()
plot_cov(cov_matrices, systematische_fout=True)
