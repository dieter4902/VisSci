import numpy as np
import matplotlib.image as mpimage
from mog import magnitude_of_gradients
import time


def seam_carve(image, seam_mask):
    """
    Removes a seam from the image depending on the seam mask. Returns an image
     that has one column less than <image>

    :param image:
    :param seam_mask:
    :return: smaller image
    """
    shrunken = image[seam_mask].reshape((image.shape[0], -1, image[..., None].shape[2]))
    return shrunken.squeeze()


def update_global_mask(global_mask, new_mask):
    """
    Updates the global_mask that contains all previous seams by adding the new path contained in new_mask.

    :param global_mask: The global mask (bool-Matrix) where the new path should be added
    :param new_mask: The mask (bool-Matrix) containing a new path
    :return: updated global Mask
    """
    reduced_idc = np.indices(global_mask.shape)[:, ~global_mask][:, new_mask.flat]
    seam_mask = np.ones_like(global_mask, dtype=bool)
    seam_mask[reduced_idc[0], reduced_idc[1]] = False
    return seam_mask


def calculate_accum_energy(energy):
    """
    Function computes the accumulated energies

    :param energy: ndarray (float)
    :return: ndarray (float)
    """
    # 2.1 Initialisieren Sie das neue resultierende Array
    # Codebeispiel:
    accum = np.array(energy)
    ...
    # 2.2 Füllen Sie das Array indem Sie die akkumulierten
    # Energien berechnen (dynamische Programmierung)
    for i in range(1, len(energy)):
        for j in range(len(energy[0])):
            accum[i, j] += np.min(accum[i - 1, j - int(j != 0):j + 2])
    # 2.3 Returnen Sie die die akkumulierten Energien
    return accum


def create_seam_mask(accumE):
    """
    Creates and returns boolean matrix containing zeros (False) where to remove the seam

    :param accumE: ndarray (float)
    :return: ndarray (bool)
    """
    # 3.1.1 Initialisieren Sie eine Maske voller True-Werte
    # Codebeispiel:

    Mask = np.ones(accumE.shape, dtype=bool)
    ...
    # 3.1.2 Finden Sie das erste Minimum der akkumulierten Energien.
    # Achtung! Nach welchem Minimum ist gefragt? Wo muss nach dem Minimum gesucht werden?

    # 3.1.3 Setzten Sie die entsprechende Stelle (np.argmin) in der Maske auf False
    pos = accumE[-1].argmin()
    Mask[-1, pos] = False
    for i in range(2, len(accumE) + 1):
        tmp = accumE[-i, pos - int(pos != 0):pos + 2].argmin()
        pos += tmp - int(pos != 0)
        Mask[-i, pos] = False
    # 3.1.4 Wiederholen Sie das für alle Zeilen von unten nach oben.

    # Codebeispiel: for row in reversed(range(0, accumE.shape[0])):

    # Achtung! Wieder: Wo muss nach dem nächsten Minimum gesucht werden?
    # Denkt dran, die Minimums müssen benachbart sein. Das schränkt die Suche
    # nach dem nächsten Minimum enorm ein.

    # 3.1.5 Returnen Sie die fertige Maske
    return Mask


def carve(path, number_of_seams_to_remove):
    # --------------------------------------------------------------------------
    # Initalisierung
    # --------------------------------------------------------------------------
    # lädt das Bild
    img = mpimage.imread('bilder/{}.jpg'.format(path))  # 'bilder/bird.jpg')
    # erstellt eine globale Maske
    # In der Maske sollen alle Pfade gespeichert werden die herrausgeschnitten wurden
    # An Anfang ist noch nichts herrausgeschnitten, also ist die Maske komplett False
    global_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

    # Parameter einstellen:
    # Tipp: hier number_of_seams_to_remove am Anfang einfach mal auf 1 setzen
    # erstellet das neue Bild, welches verkleinert wird
    new_img = np.array(img, copy=True)
    copy_img = np.array(img, copy=True)

    # --------------------------------------------------------------------------
    # Der Algorithmus
    # --------------------------------------------------------------------------
    # Für jeden Seam, der entfernt werden soll:
    for idx in range(number_of_seams_to_remove):
        # Aufgabe 1:
        # 1.1 Berechnen Sie die Gradientenlängen des Eingabe Bildes
        # und nutzen Sie diese als Energie-Werte. Sie können dazu Ihre Funktion
        # aus Aufgabe 1. (mog.py) nutzen. Dazu müssen Sie oben Ihr Skript einfügen:
        # Codebeispiel: from mog import magnitude_of_gradients
        #               energy = magnitude_of_gradients(new_img)
        # Tipp: Als Test wäre eine einfache Matrix hilfreich:
        energy = magnitude_of_gradients(new_img)

        # Aufgabe 2:

        # 2.1 Implementieren Sie die Funktion calculate_accum_energy.
        # Sie soll gegeben eine Energy-Matrix die akkumulierten Energien berechnen.
        # Codebeispiel:
        accumE = calculate_accum_energy(energy)
        # Aufgabe 3:
        # 3.1 Implementieren Sie die Funktion create_seam_mask.
        # Sie soll gegeben einer akkumulierten Energie-matrix einen Pfad finden,
        # der entfernt werden soll. Der Pfad wird mithilfe einer Maske gespeichert.
        # Beispiel:
        #     Bild                         Maske
        # |. . . / . .|     [[True, True, True, False, True, True],
        # |. . / . . .| --> [True, True, False, True, True, True],
        # |. . \ . . .|     [True, True, False, True, True, True],
        # |. . . \ . .|     [True, True, True, False, True, True]]
        #       Seam
        # Codebeispiel:
        seam_mask = create_seam_mask(accumE)

        # Aufgabe 4:
        # 4.1 Entfernen Sie den "seam" aus dem Bild mithilfe der Maske und
        # der Funktion seam_carve. Diese Funktion ist vorgegeben und muss nicht
        # implementiert werden.
        # Codebeispiel:

        new_img = seam_carve(new_img, seam_mask)
        # Aufgabe 5:
        # 5.1 Updaten Sie die globale Maske mit dem aktuellen Seam (update_global_mask).
        global_mask = update_global_mask(global_mask, seam_mask)
        # 5.2 Kopieren Sie das Originalbild und färben Sie alle Pfade, die bisher entfert wurden, rot mithilfe der globalen Maske
        # Codebeispiel:
        copy_img[global_mask, :] = [255, 0, 0]
        # Aufgabe 6:
        # 6.1 Speichere das verkleinerte Bild
        mpimage.imsave("smallerImg/{}{}.png".format(path, idx), new_img)
        # 6.2 Speichere das Orginalbild mit allen bisher entfernten Pfaden
        # 6.3 Gebe die neue Bildgröße aus:
        # print(idx, " image carved:", new_img.shape)

    mpimage.imsave("removedPaths/{}.png".format(path), copy_img)
    return new_img


'''
exakt gleiche methode wie carve() nur mit dem unterschied, dass mog nicht jedes mal neu generiert wird, sondern immernur ein mal
'''


def betterCarve(path, number_of_seams_to_remove):
    img = mpimage.imread('bilder/{}.jpg'.format(path))
    global_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    new_img = np.array(img, copy=True)
    copy_img = np.array(img, copy=True)
    energy = magnitude_of_gradients(new_img)
    for idx in range(number_of_seams_to_remove):
        accumE = calculate_accum_energy(energy)
        seam_mask = create_seam_mask(accumE)
        new_img = seam_carve(new_img, seam_mask)
        energy = seam_carve(energy, seam_mask)
        global_mask = update_global_mask(global_mask, seam_mask)
        copy_img[global_mask, :] = [255, 0, 0]
        mpimage.imsave("smallerImg/better{}{}.png".format(path, idx), new_img)
    mpimage.imsave("removedPaths/better{}.png".format(path), copy_img)
    return new_img

def benchmarkMethod(method, number_of_seams_to_remove, images):
    start = time.time()
    for path in images:
        # 6.4. Speichere das resultierende Bild nocheinmal extra.
        mpimage.imsave("final/{}{}.png".format(method.__name__, path), method(path, number_of_seams_to_remove))
    end = time.time()
    print("{}: ".format(method.__name__), end - start, " seconds")


# ------------------------------------------------------------------------------
# Main Bereich
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    number_of_seams_to_remove = 20
    images = ["tower", "bird"]
    benchmarkMethod(carve, number_of_seams_to_remove, images)
    benchmarkMethod(betterCarve, number_of_seams_to_remove, images)
    '''benchmark bei 20 seams
    carve:  35.5350399017334  seconds
    betterCarve:  11.044658422470093  seconds'''
