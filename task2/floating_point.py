import numpy as np
from math import sqrt, fabs, pi
import matplotlib.pyplot as plt

# Aufgabe 1 (a)
# i ist hier die Anzahl der Iterationen
# In jeder Iteration soll ein epsilon auf 1.0 addiert werden und mit der
# Floating-Point Darstellung von np.float64(1) bzw. np.float(32) verglichen werden.
# Starten Sie dabei mit Epsilon=1.0 und halbieren Sie den Wert in jeder Iteration (wie an der Ausgabe 2^(-i) zu sehen)
# Stoppen Sie die Iterationen, wenn np.float32(1) + epsi != np.float32(1) ist.
# Hinweis: Ja - in diesem Fall dürfen Sie Floating-Point Werte vergleichen ;)

epsi = np.float64(1)
i = 0

# Print Anweisung vor dem Loop
print('i | 2^(-i) | 1 + 2^(-i) ')
print('----------------------------------------')

##
while np.float32(1) + epsi != np.float32(1):
    print(epsi)
    epsi /= 2
    i += 1
##

# Print Anweisung in / nach dem Loop
print('{0:4.0f} | {1:16.8e} | ungleich 1'.format(i, epsi))

# Aufgabe 1 (b)
# Werten Sie 30 Iterationen aus und speichern Sie den Fehler in einem
# Fehlerarray err
N = 30
err = []
# sqrt(2) kann vorberechnet werden
sn = sqrt(2)

for n in range(2, N):
    # 1. Umfang u berechnen
    u = sn * 2 ** n
    sn = sqrt(2 - sqrt(4 - sn ** 2))
    # 2. Fehler en berechnen und in err speichern
    en = abs(2 * pi - u)
    err.append(en)
    # Fehler ausgeben
    print('{0:2d}\t{1:1.20f}\t{2:1.20e}'.format(n, u, en))

# Plotten Sie den Fehler
plt.figure(figsize=(6.0, 4.0))
plt.semilogy(range(2, N), err, 'rx')
plt.xlim(2, N - 1)
plt.ylim(1e-16, 10)
plt.show()
# auf dem generierten plot kann man beobachten, wie der errechnete umfang immer nächer an den wirklichen kreisumfang kommt.
# rein mathematisch sollte dies weiter zutreffen, aber ab einem bestimmten punkt wird en wieder größer, da die errechnete zahl an genauigkeit verliert.

# Aufgabe 1 (c)
# Löschen des Arrays und wir fangen mit der Berechnung von vorn an.
# Nur diesmal mit der leicht veranderten Variante
err = []
sn = sqrt(2)
for n in range(2, N):
    # 1. Umfang u berechnen
    u = sn * 2 ** n
    sn = sn / sqrt(2 + sqrt(4 - sn ** 2))
    # 2. Fehler en berechnen und in err speichern
    en = abs(2 * pi - u)
    err.append(en)
    # Fehler ausgeben
    print('{0:2d}\t{1:1.20f}\t{2:1.20e}'.format(n, u, en))

plt.figure(figsize=(6.0, 4.0))
plt.semilogy(range(2, N), err, 'rx')
plt.xlim(2, N - 1)
plt.ylim(1e-16, 10)
plt.show()
# dieser Plot zeigt, dass durch die berechnung des umfangs mit der verbesserten Funktion die genauigkeit länger erhalten bleibt,
# im vergleich zur vorherigen Methode
