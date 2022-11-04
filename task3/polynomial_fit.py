import numpy as np
import matplotlib.pyplot as plt
import gc

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
for i in range(5):
    b = np.load("data/d{}.npy".format(i)).reshape(200)
    for j in range(1, 21):
        x = np.linspace(-2, 2, 200)
        A = create_matrix(x, j)
        AtA = A.T.dot(A)
        Atb = A.T.dot(b)
        d = np.linalg.solve(AtA, Atb)
        plt.plot(x, b, 'r.')
        plt.plot(x, f(x, d), 'b-')
        plt.title("{}, {}".format(i, j))
        plt.show()
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
