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
               [0, SIGMA_THETA ** 2 + 0.01 ** 2, 0],
               [0, 0.01 ** 2, SIGMA_PHI ** 2 + 0.01 ** 2]])


# Laadt de dataset in als een numpy array. Hierna wordt de data getransponeerd. Dit wordt gedaan zodat alle
# r, theta en phi waarden dan in dezelfde subarray zitten. r = data[0], theta = data[1] en phi = data[2].
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


def afgeleide_r(r, theta, phi):
    """
    x, y en z afleiden naar r.
    :param r: een 504 x 1 array met hierin alle waarden voor r
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


def matrix_vermenigvuldiging(r, theta, phi):
    """
    Deze functie implementeert de formule voor de foutenpropagatie als matrix vermenigvuldiging.
    Eerst wordt een array A gecreëerd. Deze array A zal een 3 x 3 x 504 array zijn.
    Idien men A transponeerd wordt dit een 504 x 3 x 3 array waar elke 3 x 3 matrix in deze array de volgende matrix
    voorstelt:
    M = [[dx/dr, dx/dt, dx/dp]
        [dy/dr, dy/dt, dy/dp]
        [dz/dr, dz/dt, dz/dp]]

    We vermenigvuldigen deze matrix M met CR en dan met zijn getransponeerde M.T
    Dit gebeurd voor elke 3x3 matrix in de 504 x 3 x 3 array, zodat we weer een 504 x 3 x 3 array terug krijgen
    waarbij elke 3 x 3 matrix de covariantiematrix CX van die coördinaat voorstelt.

    Al deze bewerkingen worden via een einsum gedaan aangezien dit veel tijd bespaart.

    Ten slotte pickelen we de geresulteerde array. Hierdoor kan hij makkelijk weer opgeroepen worden om te gebruiken
    tijdens het plotten zonder dat hij opnieuw berekend moet worden.

    :param r: een 504 x 1 array met hierin alle waarden voor r
    :param theta: een 504 x 1 array met hierin alle waarden voor theta
    :param phi: een 504 x 1 array met hierin alle waarden voor phi
    :return:
    """
    A = np.array(([afgeleide_r(r, theta, phi),
                   afgeleide_theta(r, theta, phi),
                   afgeleide_phi(r, theta, phi)]))

    CX = np.einsum('kji, kl, lmi -> ijm', A, CR, A)
    """
    Breakdown einsum:
    Transponeer A:
    np.einsum(kji) of np.einsum(ijk -> kji)
    
    Hierein is A al getransponeerd
    i is de as waarlangs de 3x3 matrices staan.
    dus als X die 3x3 matrices zijn is een element hiering X_jk
    We vermenigvuldigen elke van deze X matrix mat CR jk, kl wat betekent dat we de rij van X met de kolom van CR
    vermenigvuldigen, i.e. matrixvermenigvuldiging.
    
    het resulterende van deze eerste vermenigvuldiging zal dan ijl zijn. Dan vermenigvuldigen we deze l matrix.
    de derde matrix zou eigenlijk ilm zijn, maar deze zal de getransponeerde van de interne matrix van A moeten zijn.
    ilm wordt dus iml wat deze interne matrix vermenigvuldigd of:  np.einsum(iml) of np.einsum(ilm -> iml).
    Dus vermenigvuldigen we opniew de rij met de kolom van deze getransponeerde. Dat is wat de volgende betekent.
    CX = np.einsum('ijk, kl, iml -> ijm', A, CR, A) 
    
    Indien we dit alles samen zetten krijgen we:
    CX = np.einsum('kji, kl, lmi -> ijm', A, CR, A)
    """

    """
    performance boost with einsum:
    np.einsum('kji, kl, lmi -> ijm', A, CR, A) -> 0.00010899999999969268 s
    A = A.T & CX = np.einsum('ijk, kl, iml -> ijm', A, CR, A) -> 0.00017989999999983958 s
    for loop -> 0.002914499999999931 s
    """

    outfile = open("covariantiematrix_geen_correlaties", 'wb')
    pickle.dump(CX, outfile)
    outfile.close()


def plot_coord_2d(x, y, assen, color):
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
                        mpatches.Patch(color='blue', label='z')
                        ], loc='upper right')

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


def plot_fouten(x_waarde, y_waardes, systematische_fout, spherische_coord):
    plt.scatter(x_waarde, y_waardes[0], marker='.', color='royalblue', label='x')
    plt.scatter(x_waarde, y_waardes[1], marker='.', color='tab:green', label='y')
    plt.scatter(x_waarde, y_waardes[2], marker='.', color='darkorange', label='z')
    plt.xlabel('{} coördinaten'.format(spherische_coord)), plt.ylabel('x, y en z coördinaten')
    plt.title('De fouten van de x, y en z coordinaten in functie van {}.'.format(spherische_coord))
    plt.legend()
    plt.savefig('plots/{}/fout_ifv_{}.pdf'.format('met S fout' if systematische_fout else 'zonder S fout',
                                                  spherische_coord),
                bbox_inches="tight")
    plt.clf()


def plot_correlaties(x_waarde, y_waardes, systematische_fout, spherische_coord):
    plt.scatter(x_waarde, y_waardes[0], marker='.', color='royalblue', label='xy')
    plt.scatter(x_waarde, y_waardes[1], marker='.', color='tab:green', label='yz')
    plt.scatter(x_waarde, y_waardes[2], marker='.', color='darkorange', label='zx')
    plt.xlabel('{} coördinaten'.format(spherische_coord))
    plt.ylabel('de correlaties tussen xy, yz en zx'.format(spherische_coord))
    plt.title('De correlaties tussen de xy, yz en zx in functie van {}.'.format(spherische_coord))
    plt.legend()
    plt.savefig('plots/{}/correlaties_ifv_{}.pdf'.format('met S fout' if systematische_fout else 'zonder S fout',
                                                         spherische_coord),
                bbox_inches="tight")
    plt.clf()


def plot_cov(cov_matrices, systematische_fout):
    cov_per_coordinaat = np.transpose(cov_matrices, (1, 2, 0))

    fout_x = cov_per_coordinaat[0][0]
    fout_y = cov_per_coordinaat[1][1]
    fout_z = cov_per_coordinaat[2][2]
    cov_xy = cov_per_coordinaat[0][1]
    cov_yz = cov_per_coordinaat[1][2]
    cov_zx = cov_per_coordinaat[2][0]

    corr_xy = cov_xy / np.sqrt(fout_x * fout_y)
    corr_yz = cov_yz / np.sqrt(fout_y * fout_z)
    corr_zx = cov_zx / np.sqrt(fout_z * fout_x)

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


# voor de coordinaattransformatie uit
# coordinaattransformatie()

# bereken de covariantiematrices
# matrix_vermenigvuldiging(data[0], data[1], data[2])

# plot de covariantiematrices
inputfile = open("covariantiematrix_geen_correlaties", 'rb')
cov_matrices = pickle.load(inputfile)
plot_cov(cov_matrices, systematische_fout=False)
