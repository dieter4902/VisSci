import numpy as np
import matplotlib.pyplot as plt

# Laden der gegebenen Daten d0 - d4

d0 = np.load('data/d0.npy')
data = np.zeros([5], dtype=object)
for i in range(5):
    data[i] = np.load("data/d{}.npy".format(i))
# fig, ax = plt.subplots(2, len(data))
# for i, points in enumerate(data):
# ax[0, i].plot(points)
# plt.show()

x = np.linspace(-2, 2, 200)


# Implementieren Sie ein Funktion, die gegeben den x-Werten und dem Funktiongrad
# die Matrix A aufstellt.

# for i, points in enumerate(data):
# A = np.column_stack((x ** 2, x, np.ones_like(x)))
# coeff = np.linalg.lstsq(A, points, rcond=None)[0]
# Darstellung der experimentellen Daten
# Darstellung der Ausgleichsgeraden
# ax[1, i].plot(x, points, 'r.')
# ax[1, i].plot(x, coeff[0] * x ** 2 + coeff[1] * x + coeff[2], 'b-')
# ax[1, i].plot(x, np.exp(x), 'g--')


# plt.show()


def create_matrix(x, degree):
    a = []
    for i in range(degree + 1):
        a.append(x ** (degree - i))
        # print(x ** (degree - i))
    return np.array(a).T


def f(x_cord, a):
    result = 0.0
    for n, element in enumerate(a):
        result += x_cord * (element ** (i - n))
    return result


# Lösen Sie das lineare Ausgleichsproblem
# Hinweis: Nutzen Sie bitte hier nicht np.linalg.lstsq!, sondern implementieren sie A^T A x = A^T b selbst
for b in data:
    for i in range(1, 21):
        A = create_matrix(x, i)
        AtA = A.T.dot(A)
        Atb = A.T * b
        d = np.linalg.solve(AtA, Atb)
        plt.plot(x, b, 'r.')
        plt.plot(x, f(x, d), 'b-')
        plt.title(i)
        plt.show()

# Stellen Sie die Funktion mit Hilfe der ermittelten Koeffizienten mit matplotlib
# np.poly1d
