import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Laden der gegebenen Daten d0 - d4


# obszolet gemacht durch unten folgenden code
data = np.zeros([5], dtype=object)
for i in range(5):
    data[i] = np.load("data/d{}.npy".format(i)).reshape(200)


# Implementieren Sie ein Funktion, die gegeben den x-Werten und dem Funktiongrad
# die Matrix A aufstellt.
def create_matrix(n, degree):
    a = []
    for i in range(degree + 1):
        a.append(n ** (degree - i))
    return np.array(a).T


def f(x_cord, a):
    results = []
    for cord in x_cord:
        result = 0.0
        for n, element in enumerate(a):
            result += element * (np.power(cord, (len(a) - n - 1)))
        results.append(result)

    return np.array(results)


# Lösen Sie das lineare Ausgleichsproblem
# Hinweis: Nutzen Sie bitte hier nicht np.linalg.lstsq!, sondern implementieren sie A^T A x = A^T b selbst
x = np.linspace(-2, 2, 200)
file = open('output.txt', 'w')
file.flush()
fig, ax = plt.subplots(nrows=2, ncols=5)
fig.subplots_adjust(hspace =0.5, wspace=0.5)
for i in range(5):
    b = np.load("data/d{}.npy".format(i)).reshape(200)
    # color_legend = []
    # for k in range(20):
    #    color_legend.append(
    #        mlines.Line2D([], [], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][k % 10], marker='s', ls='',
    #                      label=k + 1))

    ax[0][i].plot(x, b, 'r.')
    for j in range(1, 21):
        # plt.legend(handles=color_legend)
        A = create_matrix(x, j)
        AtA = A.T.dot(A)
        Atb = A.T.dot(b)
        d = np.linalg.solve(AtA, Atb)
        error = np.sum(np.absolute(f(x, d) - b))
        ax[1][i].plot(j, error, '.')
        ax[0][i].plot(x, f(x, d), '-')
        file.write("dataset no:{}, polynomial no:{}, error:{}, polynomial:\n{}\n".format(i, j, error, np.poly1d(d)))
    ax[0][i].title.set_text("Dataset\nn°{}".format(i))
    ax[1][i].title.set_text("Residual\nn°{}".format(i))
plt.savefig('output.png', dpi=600)
file.close()
# Stellen Sie die Funktion mit Hilfe der ermittelten Koeffizienten mit matplotlib
# np.poly1d
# A = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
# b = np.array([[2], [1], [2], [2.5]])
# AtA = A.T.dot(A)
# Atb = A.T.dot(b)
# x = np.linalg.solve(AtA, Atb)
# ptx = np.array([1, 2, 3, 4])
# plt.plot([1, 2, 3, 4], [2, 1, 2, 2.5], 'ro')
# plt.plot(ptx, f(ptx, x))
# plt.show()
