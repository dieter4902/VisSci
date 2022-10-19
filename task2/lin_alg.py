import numpy as np
import math

# (a) Berechnen Sie den Winkel $\alpha$ in Grad zwischen den folgenden beiden Vektoren $a=[1.,1.77]$ und $b=[1.5,1.5]$
a = np.array([-1., 1.77])
b = np.array([1.5, 1.5])

# YOUR CODE HERE
alpha = math.degrees(math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
print(alpha)

# (b) Gegeben ist die quadratische regulaere Matrix A und ein Ergbnisvektor b. Rechnen Sie unter Nutzung der Inversen die Loesung x des Gleichungssystems Ax = b aus.
# YOUR CODE HERE
mA = np.array([[2, 3, 4], [3, -2, -1], [5, 4, 3]])
vB = np.array([1.4, 1.2, 1.4])
mA_inv = np.linalg.inv(mA)
x = mA_inv.dot(vB)
print(x)
print(mA.dot(x))


# (c) Schreiben Sie eine Funktion die das Matrixprodukt berechnet. Nutzen Sie dafür nicht die Numpy Implementierung.
# Hinweis: Fangen Sie bitte mögliche falsche Eingabegroessen in der Funktion ab und werfen einen AssertionError
# assert Expression[, Arguments]

def matmult(M1, M2):
    # YOUR CODE HERE
    res = np.empty(M1.shape)
    n = 0
    for a1 in M1:
        o = 0
        for b1 in M2.transpose():
            m = 0
            for i in a1:
                # print("%d * %f" % (i, b1[m]))
                res[n, o] += i * b1[m]
                m += 1
            o += 1
        n += 1
    return res


M1 = np.array([[1, 2], [3, 4], [5, 6]])
M2 = np.array([[2, 0], [0, 2]])

print(M2.transpose())
# print(M1)
# print(M2)

# print(np.matmul(M1, M2))
M_res = matmult(M1, M2)
print(M_res)
