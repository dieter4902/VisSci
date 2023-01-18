import numpy as np
import matplotlib.pyplot as plt
import helper
import os
import plotly

np.random.seed(1)


class TwoLayerNeuralNetwork:
    """
    Ein 2-Layer 'fully-connected' neural network, d.h. alle Neuronen sind mit allen anderen
    verbunden. Die Anzahl der Eingabevektoren ist N mit einer Dimension D, einem 'Hidden'-Layer mit
    H Neuronen. Es soll eine Klassifikation über 10 Klassen (C) durchgeführt werden.
    Wir trainieren das Netzwerk mit einer Kreuzentropie-Loss Funktion. Das Netzwerk nutzt ReLU 
    Aktivierungsfunktionen nach dem ersten Layer.
    Die Architektur des Netzwerkes läßt sich abstrakt so darstellen:
    Eingabe - 'fully connected'-Layer - ReLU - 'fully connected'-Layer - Softmax

    Die Ausgabe aus dem 2.Layer sind die 'Scores' (Wahrscheinlichkeiten) für jede Klasse.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Intitialisierung des Netzes - Die Gewichtungsmatrizen und die Bias-Vektoren werden mit
        Zufallswerten initialisiert.
        W1: 1.Layer Gewichte (D, H)
        b1: 1.Layer Bias (H,)
        W2: 2.Layer Gewichte (H, C)
        b2: 2.Layer Bias (C,)

        :param input_size: Die CIFAR-10 Bilder haben die Dimension D (32*32*3).
        :param hidden_size: Anzahl der Neuronen im Hidden-Layer H.
        :param output_size: Anzahl der Klassen C.
        :param std: Skalierungsfaktoren für die Initialisierung (muss klein sein)
        :return:
        """
        self.W1 = std * np.random.randn(input_size, hidden_size)
        self.b1 = std * np.random.randn(1, hidden_size)
        self.W2 = std * np.random.randn(hidden_size, output_size)
        self.b2 = std * np.random.randn(1, output_size)

    def softmax(self, z):
        z -= np.max(z)
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def relu(self, x):
        return np.maximum(0.0, x)

    def relu_derivative(self, output):
        output[output <= 0] = 0
        output[output > 0] = 1
        return output

    def loss_deriv_softmax(self, activation, y_batch):
        """ 
        derivative of the mean cross entropy loss function, that includes the derivate of the softmax
        for further explanations see here: https://deepnotes.io/softmax-crossentropy
        """

        batch_size = y_batch.shape[0]
        dCda2 = activation
        dCda2[range(batch_size), y_batch] -= 1
        dCda2 /= batch_size
        return dCda2

    def loss_crossentropy(self, activation, y_batch):
        """
        Berechnet den loss (Fehler) des 2-Layer-Netzes

        :param activation: Aktivierungen / Output des Netzes
        :param y_batch: Vektor mit den Trainingslabels y[i] für einen Batch enthält ein Label aus X[i] 
                        und jedes y[i] ist ein Integer zwischen 0 <= y[i] < C (Anzahl der Klassen),
        :return: loss - normalisierter Fehler des Batches
        """

        batch_size = y_batch.shape[0]
        correct_logprobs = -np.log(activation[range(batch_size), y_batch])
        loss = np.sum(correct_logprobs) / batch_size
        return loss

    def forward(self, X, y):
        """
        Führt den gesamten Forward Prozess durch und berechnet den Fehler (loss) und die Aktivierungen a1 und
        a2 und gibt diese Werte zuruück
        :param X: Trainings bzw. Testset
        :param y: Labels des Trainings- bzw. Testsets
        :return: loss, m1, a1, a2
        """

        # Berechen Sie den score
        # Berechnen Sie den Forward-Schritt und geben Sie den Vektor mit Scores zurueck
        # Nutzen Sie die ReLU Aktivierungsfunktion im ersten Layer
        m1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(m1)
        a2 = self.softmax(np.dot(a1, self.W2) + self.b2)
        # Berechnen Sie die Klassenwahrscheinlichkeiten unter Nutzung der softmax Funktion

        # Berechnen Sie den Fehler mit der cross-entropy Funktion

        loss = self.loss_crossentropy(a2, y)
        return loss, m1, a1, a2

    def backward(self, m1, a1, a2, X, y):
        """
        Backward pass- dabei wird der Gradient der Gewichte W1, W2 und der Biases b1, b2 aus den Ausgaben des Netzes
        berechnet und die Gradienten der einzelnen Layer als ein Dictionary zurückgegeben.
        Zum Beispiel sollte grads['W1'] die Gradienten von self.W1 enthalten (das ist eine Matrix der gleichen Größe
        wie self.W1.
        :param m1: Aktivierung aus dem 2.Layer vor Aktivierungsfunktion
        :param a1: Aktivierung aus dem 1.Layer
        :param a2: Aktivierung aus dem 2.Layer -> Output des Netzes
        :param X:
        :param y:
        :return gradienten dictionaty :
        """

        # Füllen Sie das Dictionary grads['W2'], grads['b2'], grads['W1'], grads['b1']
        grads = {'W1': None, 'b1': None, 'W2': None, 'b2': None}

        # Nutzen Sie dabei die Notizen aus der Vorlesung und die gegebenen Ableitungsfunktionen
        dc_da2 = self.loss_deriv_softmax(a2, y)
        dm2_da1 = self.W2
        da1_dm1 = self.relu_derivative(m1)
        dm1_dw1 = X
        dm2_dw2 = a1

        tmp1 = dc_da2 @ dm2_da1.T
        tmp2 = tmp1 * da1_dm1

        # Backward pass: Berechnen Sie die Gradienten
        grads['W1'] = np.dot(dm1_dw1.T, tmp2)
        grads['W2'] = np.dot(dm2_dw2.T, dc_da2)
        grads['b1'] = np.sum(tmp2, axis=0)
        grads['b2'] = np.sum(dc_da2, axis=0)

        return grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95, num_iters=100,
              batch_size=50, verbose=False):
        """
        Training des Neuronalen Netzwerkes unter der Nutzung des iterativen
        Optimierungsverfahrens Stochastic Gradient Descent
        Train this neural network using stochastic gradient descent.

        :param X: Numpy Array der Größe (N,D)
        :param y: Numpy Array der Größe (N,) mit den jeweiligen Labels y[i] = c. Das bedeutet, dass X[i] das label c hat
                  mit 0 <= c < C
        :param X_val: Numpy Array der Größe (N_val,D) mit den Validierungs-/Testdaten
        :param y_val: Numpy Array der Größe (N_val,) mit den Labels für die Validierungs-/Testdaten
        :param learning_rate: Faktor der Lernrate für den Optimierungsprozess
        :param learning_rate_decay: gibt an, inwieweit die Lernrate in jeder Epoche angepasst werden soll
        :param num_iters: Anzahl der Iterationen der Optimierung
        :param batch_size: Anzahl der Trainingseingabebilder, die in jedem forward-Schritt mitgegeben werden sollen
        :param verbose: boolean, ob etwas ausgegeben werden soll
        :return: dict (fuer die Auswertung) - enthält Fehler und Genauigkeit der Klassifizierung für jede Iteration bzw. Epoche
        """
        num_train = X.shape[0]
        iterations_per_epoch = int(max(num_train / batch_size, 1))

        # Wir nutzen einen Stochastischen Gradient Decent (SGD) Optimierer um unsere
        # Parameter W1,W2,b1,b2 zu optimieren
        loss_history = []
        loss_val_history = []
        train_acc_history = []
        val_acc_history = []

        sample_propabilities = np.ones(X.shape[0])
        for it in range(num_iters):
            mask = np.random.choice(X.shape[0], batch_size)
            X_batch = X[mask]
            y_batch = y[mask]
            # Erzeugen Sie einen zufälligen Batch der Größe batch_size
            # aus den Trainingsdaten und speichern diese in X_batch und y_batch

            # Berechnung von loss und gradient für den aktuellen Batch

            # Merken des Fehlers für den Plot
            # loss_history.append(loss)
            # Berechnung des Fehlers mit den aktuellen Parametern (W, b)
            # mit dem Testset

            loss, m1, a1, a2 = self.forward(X_batch, y_batch)
            grads = self.backward(m1, a1, a2, X_batch, y_batch)
            loss_history.append(loss)
            loss_val_history.append(self.forward(X_val, y_val)[0])

            # Nutzen Sie die Gradienten aus der Backward-Funktion und passen
            # Sie die Parameter an (self.W1, self.b1 etc). Diese werden mit der Lernrate
            # gewichtet

            self.W1 -= learning_rate * grads['W1']
            self.b1 -= learning_rate * grads['b1']
            self.W2 -= learning_rate * grads['W2']
            self.b2 -= learning_rate * grads['b2']

            # Ausgabe des aktuellen Fehlers. Diese sollte am Anfang erstmal nach unten gehen
            # kann aber immer etwas schwanken.
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Wir überprüfen jede Epoche die Genauigkeit (von Trainingsset und Testset)
            # und dämpfen die Lernrate
            if it % iterations_per_epoch == 0:
                # Überprüfung der Klassifikationsgenauigkeit
                train_acc = (self.predict(X) == y).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                print('epoch done... acc', val_acc)

                # Dämpfung der Lernrate
                learning_rate *= learning_rate_decay

        # Zum Plotten der Genauigkeiten geben wir die
        # gesammelten Daten zurück
        return {
            'loss_history': loss_history,
            'loss_val_history': loss_val_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Benutzen Sie die trainierten Gewichte des 2-Layer-Netzes um die Klassen für das
        Validierungsset vorherzusagen. Dafür müssen Sie für das/die Eingabebilder X nur
        die Scores berechnen. Der höchste Score ist die vorhergesagte Klasse. Dieser wird in y_pred
        geschrieben und zurückgegeben.

        :param X: Numpy Array der Größe (N,D)
        :return: y_pred Numpy Array der Größe (N,) die die jeweiligen Labels für alle Elemente in X enthaelt.
        y_pred[i] = c bedeutet, das fuer X[i] die Klasse c mit 0<=c<C vorhergesagt wurde
        """

        a1 = self.relu(np.dot(X, self.W1) + self.b1)
        a2 = self.softmax(np.dot(a1, self.W2) + self.b2)
        y_pred = np.argmax(a2, axis=1)
        """
        hidden_layer = self.relu(np.dot(X, self.W1) + self.b1)
        scores = np.dot(hidden_layer, self.W2) + self.b2
        y_pred = np.argmax(scores, axis=1)"""
        return y_pred


if __name__ == '__main__':
    X_train, y_train, X_val, y_val = helper.prepare_CIFAR10_images()
    # Laden der Bilder. Hinweis - wir nutzen nur die Trainigsbilder zum Trainieren und die Validierungsbilder zum Testen
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)

    # Grösse der Bilder
    input_size = 32 * 32 * 3
    # Anzahl der Klassen
    num_classes = 10

    #############################################
    # Hyperparameter
    #############################################

    # mit diesen Parametern sollten Sie in etwa auf eine
    # Klassifikationsgenauigkeit von 43% kommen. Optimieren Sie die
    # Hyperparameter um die Genauigkeit zu erhöhen (bitte tun sie das
    # systematisch und nicht einfach durch probieren - also z.B. in einem
    # for-loop eine Reihe von Parametern testen und die Einzelbilder abspeichern)

    hidden_size = 250  # Anzahl der Neuronen im Hidden Layer
    num_iter = 4000  # Anzahl der Optimierungsiterationen
    batch_size = 300  # Eingabeanzahl der Bilder
    learning_rate = 0.001  # Lernrate
    learning_rate_decay = 0.95  # Lernratenabschwächung
    """
    net = TwoLayerNeuralNetwork(input_size, hidden_size, num_classes)  # reset network
    stats = net.train(X_train, y_train, X_val, y_val,
                      num_iters=num_iter, batch_size=batch_size,
                      learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
                      verbose=False)

    print('Final training loss: ', stats['loss_history'][-1])
    print('Final validation loss: ', stats['loss_val_history'][-1])
    print('Final validation accuracy: ', stats['val_acc_history'][-1])

    helper.plot_net_weights(net)
    helper.plot_accuracy(stats)
    helper.plot_loss(stats)
    """
    data = np.array([[0, 0, 0, 0, 0, 0]])
    """
    for lr in range(3000, 4500, 500):
        learning_rate = ...
        for ld in range(3000, 4500, 500):
            learning_rate_decay = ...
            """
    for num_iter in range(3000, 4500, 500):
        for hidden_size in range(50, 300, 50):
            for batch_size in range(200, 500, 50):
                net = TwoLayerNeuralNetwork(input_size, hidden_size, num_classes)  # reset network
                stats = net.train(X_train, y_train, X_val, y_val,
                                  num_iters=num_iter, batch_size=batch_size,
                                  learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
                                  verbose=False)
                print('Final validation accuracy: ', stats['val_acc_history'][-1])
                data = np.vstack([data, [hidden_size, num_iter, batch_size, learning_rate, learning_rate_decay,
                                         stats['val_acc_history'][-1]]])

    fig = plotly.graph_objs.Figure(data=
    plotly.graph_objs.Parcoords(
        line_color='blue',
        dimensions=list([
            dict(label='hidden_size', values=data[1:, 0]),
            dict(label='num_iter', values=data[1:, 1]),
            dict(label='batch_size', values=data[1:, 2]),
            dict(label='learning_rate', values=data[1:, 3]),
            dict(label='learning_rate_decay', values=data[:, 4]),
            dict(label='accuracy', values=data[1:, 5])
        ])
    )
    )
    fig.show()
