import numpy as np
import lib
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimage


####################################################################################################


def load_images(path: str) -> list:
    """
    Load all images in path

    :param path: path of directory containing image files

    :return images: list of images (each image as numpy.ndarray and dtype=float64)
    """
    # 1.1 Laden Sie für jedes Bild in dem Ordner das Bild als numpy.array und
    # speichern Sie es in einer "Datenbank" eine Liste.
    # Tipp: Mit glob.glob("data/train/*") bekommen Sie eine Liste mit allen
    # Dateien in dem angegebenen Verzeichnis.

    images = []
    for image in glob.glob(path):
        images.append(plt.imread(image))
    return images

    # 1.2 Geben Sie die Liste zurück


def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    :param images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    :return D: data matrix that contains the flattened images as rows
    """
    # 2.1 Initalisiere die Datenmatrix mit der richtigen Größe und Typ.
    # Achtung! Welche Dimension hat die Matrix?
    num_images = len(images)
    image_shape = images[0].shape

    # Initialize data matrix with appropriate size and type
    D = np.empty((num_images, np.prod(image_shape)), dtype=np.float64)

    # 2.2 Fügen Sie die Bilder als Zeilen in die Matrix ein.
    for i, image in enumerate(images):
        D[i, :] = image.flatten()

    return D
    # 2.3 Geben Sie die Matrix zurück


def calculate_svd(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform SVD analysis for given data matrix.

    :param D: data matrix of size n x m where n is the number of observations and m the number of variables

    :return eigenvec: matrix containing principal components as rows
    :return singular_values: singular values associated with eigenvectors
    :return mean_data: mean that was subtracted from data
    """
    # 3.1 Berechnen Sie den Mittelpukt der Daten
    # Tipp: Dies ist in einer Zeile möglich (np.mean, besitzt ein Argument names axis)
    mean_data = np.mean(D, axis=0)
    centered_data = D - mean_data
    eigenvec, singular_values, _ = np.linalg.svd(centered_data, full_matrices=False)
    # 3.2 Berechnen Sie die Hauptkomponeten sowie die Singulärwerte der ZENTRIERTEN Daten.
    # Dazu können Sie numpy.linalg.svd(..., full_matrices=False) nutzen.
    # 3.3 Geben Sie die Hauptkomponenten, die Singulärwerte sowie den Mittelpunkt der Daten zurück
    return eigenvec, singular_values, mean_data


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    :param singular_values: vector containing singular values
    :param threshold: threshold for determining k (default = 0.8)

    :return k: threshold index
    """

    # 4.1 Normalizieren Sie die Singulärwerte d.h. die Summe aller Singlärwerte soll 1 sein
    singular_values /= np.sum(singular_values)
    k = 0
    energy = 0
    while energy < threshold:
        energy += singular_values[k]
        k += 1

    return k
    # 4.2 Finden Sie den index k, sodass die ersten k Singulärwerte >= dem Threshold sind.

    # 4.3 Geben Sie k zurück


def project_faces(pcs: np.ndarray, mean_data: np.ndarray, images: list) -> np.ndarray:
    """
    Project given image set into basis.

    :param pcs: matrix containing principal components / eigenfunctions as rows
    :param images: original input images from which pcs were created
    :param mean_data: mean data that was subtracted before computation of SVD/PCA

    :return coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """
    # 5.1 Initialisieren Sie die Koeffizienten für die Basis.
    # Sie sollen als Zeilen in einem np.array gespeichert werden.
    coefficients = np.empty((len(images), pcs.shape[0]))

    # 5.1 Berechnen Sie für jedes Bild die Koeffizienten.
    # Achtung! Denkt daran, dass die Daten zentriert werden müssen.
    for i, img in enumerate(images):
        centered_img = img - np.reshape(mean_data, img.shape)
        P = np.zeros(pcs.shape)
        print(centered_img.shape)
        print(P.shape)
        P[:len(centered_img), :] += centered_img[:len(P[0]), :len(P)]
        coefficients[i, :] = np.dot(P, pcs.T)  # funktioniert alles nicht 😩

    # 5.2 Geben Sie die Koeffizenten zurück
    return coefficients


def identify_faces(coeffs_train: np.ndarray, coeffs_test: np.ndarray) -> (
    np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    :param coeffs_train: coefficients for training images, each image is represented in a row
    :param coeffs_test: coefficients for test images, each image is represented in a row

    :return indices: array containing the indices of the found matches
    """
    # 8.1 Berechnen Sie für jeden Testvektor den nächsten Trainingsvektor.
    # Achtung! Die Distanzfunktion ist definiert über den Winkel zwischen den Vektoren.
    indices = np.empty(coeffs_test.shape[0], dtype=int)
    for i, coeffs_test_img in enumerate(coeffs_test):
        angles = np.arccos(np.sum(coeffs_test_img * coeffs_train, axis=1) / (
                np.linalg.norm(coeffs_test_img) * np.linalg.norm(coeffs_train, axis=1)))
        index = np.argmin(angles)
        indices[i] = index
    return indices


if __name__ == '__main__':
    ...
    # 1. Aufgabe: Laden Sie die Trainingsbilder.
    # Implementieren Sie dazu die Funktion load_images.
    train_img = load_images("data/train/*")

    # 2. Aufgabe: Konvertieren Sie die Bilder zu Vektoren die Sie alle übereinander speichern,
    # sodass sich eine n x m Matrix ergibt (dabei ist n die Anzahl der Bilder und m die Länge des Bildvektors).
    # Implementieren Sie dazu die Funktion setup_data_matrix.
    D = setup_data_matrix(train_img)

    # 3. Aufgabe: Finden Sie alle Hauptkomponeten des Datensatztes.
    # Implementieren Sie dazu die Funktion calculate_svd
    eigenvec, singular_values, mean_data = calculate_svd(D)

    # 4. Aufgabe: Entfernen Sie die "unwichtigsten" Basisvektoren.
    # Implementieren Sie dazu die Funktion accumulated_energy um zu wissen wie viele
    # Baisvektoren behalten werden sollen. Plotten Sie Ihr Ergebniss mittels
    # lib.plot_singular_values_and_energy
    k = accumulated_energy(singular_values)

    # 5. Aufgabe: Projizieren Sie die Trainingsdaten in den gefundenen k-dimensionalen Raum,
    # indem Sie die Koeffizienten für die gefundene Basis finden.
    # Implementieren Sie dazu die Funktion project_faces
    train_coefficients = project_faces(eigenvec[:k, :], mean_data, train_img)

    # 6. Aufgabe: Laden Sie die Test Bilder (load_images).
    test_images = load_images("data/test/*")

    # 7. Aufgabe: Projizieren Sie die Testbilder in den gefundenen k-dimensionalen Raum (project_faces).
    test_coefficients = project_faces(eigenvec[:k, :], mean_data, train_img)

    # 8. Aufgabe: Berechnen Sie für jedes Testbild das nächste Trainingsbild in dem
    # gefundenen k-dimensionalen Raum. Die Distanzfunktion ist über den Winkel zwischen den Punkten definiert.
    # Implementieren Sie dazu die Funktion identify_faces.
    match_indices = identify_faces(train_coefficients, test_coefficients)
    # Plotten Sie ihr Ergebniss mit der Funktion lib.plot_identified_faces
    lib.plot_identified_faces(match_indices, train_img, test_images, eigenvec, test_coefficients, mean_data)
    # plot the identified faces
