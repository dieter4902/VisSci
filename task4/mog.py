import numpy as np
import matplotlib.image as mpimage
import matplotlib.pyplot as plt


def show_image(img):
    """
    Shows an image (img) using matplotlib
    """
    if isinstance(img, np.ndarray):
        if img.shape[-1] == 3 or img.shape[-1] == 4:
            plt.imshow(img[..., :3])
        if img.shape[-1] == 1 or img.shape[-1] > 4:
            plt.imshow(img[:, :], cmap="gray")
        plt.show()


def convolution2D(img, kernel):
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix
    :return: result of the convolution
    """
    # 1.1.1 Initialisieren Sie das resultierende Bild
    # Codebeispiel: new_img = np.zeros(img.shape)
    k_x, k_y = np.shape(kernel)
    new_img = np.zeros(img.shape)
    padding_x = k_x // 2
    padding_y = k_y // 2
    padded_img = np.pad(img, (padding_x, padding_y), 'edge')
    x, y = np.shape(new_img)

    # 1.1.2 Implementieren Sie die Faltung.

    for r in range(padding_x, x - padding_x):  # bei 1 und nicht 0 anfangen wegen padding
        for c in range(padding_y, y - padding_y):
            new_img[r, c] = np.sum(
                kernel * padded_img[r - padding_x: r + k_x - padding_x, c - padding_y:c + k_y - padding_y])
    # Achtung: die Faltung (convolution) soll mit beliebig großen Kernels funktionieren.
    # Tipp: Nutzen Sie so gut es geht Numpy, sonst dauert der Algorithmus zu lange.
    # D.h. Iterieren Sie nicht über den Kernel, nur über das Bild. Der Rest geht mit Numpy.

    # Achtung! Achteten Sie darauf, dass wir ein Randproblem haben. Wie ist die Faltung am Rand definiert?
    # Tipp: Es gibt eine Funktion np.pad(Matrix, 5, mode="edge") die ein Array an den Rändern erweitert.

    # 1.1.3 Returnen Sie das resultierende "Bild"/Matrix
    # Codebeispiel: return newimg
    return new_img


def magnitude_of_gradients(RGB_img):
    """
    Computes the magnitude of gradients using x-sobel and y-sobel 2Dconvolution

    :param img: RGB image
    :return: length of the gradient
    """
    # 3.1.1 Wandeln Sie das RGB Bild in ein grayscale Bild um.
    gray = RGB_img[..., :3] @ np.array([0.299, 0.587, 0.114])
    # 3.1.2 Definieren Sie den x-Sobel Kernel und y-Sobel Kernel.
    x_Sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_Sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # 3.1.3 TODO: Nutzen Sie sie convolution2D Funktion um die Gradienten in x- und y-Richtung zu berechnen.

    # 3.1.4 TODO: Nutzen Sie die zwei resultierenden Gradienten um die gesammt Gradientenlängen an jedem Pixel auszurechnen.

    # Diese if Abfrage (if __name__ == '__main__':) sorgt dafür, dass der Code nur
    # ausgeführt wird, wenn die Datei (mog.py) per python/jupyter ausgeführt wird ($ python mog.py).
    # Solltet Ihr die Datei per "import mog" in einem anderen Script einbinden, wird dieser Code übersprungen.


if __name__ == '__main__':
    # Bild laden und zu float konvertieren
    img = mpimage.imread('bilder/tower.jpg')
    img = img.astype("float64")

    # Wandelt RGB Bild in ein grayscale Bild um
    gray = img[..., :3] @ np.array([0.299, 0.587, 0.114])

    # Aufgabe 1.
    # 1.1 Implementieren Sie die convolution2D Funktion (oben)

    # Aufgabe 2.
    # 2.1 Definieren Sie mindestens 5 verschiedene Kernels (darunter sollten beide Sobel sein) und testen Sie sie auf dem grayscale Bild indem Sie convolution2D aufrufen.
    # 2.2 TODO: Speichern Sie alle Resultate als Bilder (sehe Tipp 2). Es sollten 5 Bilder sein.
    # xsobel
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    plt.imshow(convolution2D(gray, kernel), cmap='gray')
    plt.title("xsobel")
    plt.show()
    # ysobel
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    plt.imshow(convolution2D(gray, kernel), cmap='gray')
    plt.title("ysobel")
    plt.show()
    # blur kernel
    kernel = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
    plt.imshow(convolution2D(gray, kernel), cmap='gray')
    plt.title("blur kernel")
    plt.show()
    # sharpen kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    plt.imshow(convolution2D(gray, kernel), cmap='gray')
    plt.title("sharpen kernel")
    plt.show()
    # identity
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    plt.imshow(convolution2D(gray, kernel), cmap='gray')
    plt.title("identity")
    plt.show()

    # Aufgabe 3:
    # 3.1 TODO: Implementieren Sie die magnitude_of_gradients Funktion (oben) und testen Sie sie mit dem RGB Bild.
    # 3.2 TODO: Speichern Sie das Resultat als Bild (sehe Tipp 2).

    # ------------------------------------------------
    # Nützliche Funktionen:
    # ------------------------------------------------
    # Tipp 1: So können Sie eine Matrix als Bild anzeigen:
    # show_image(gray)

    # Tipp 2: So können Sie eine NxMx3 Matrix als Bild speichern:
    # mpimage.imsave("test.png", img)
    # und so können Sie eine NxM Matrix als grayscale Bild speichern:
    # mpimage.imsave("test.png", gray, cmap="gray")
