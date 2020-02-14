import numpy as np

from distance import distance


def gen_centroids(dataset, points, k, features):

    centroids = np.empty(shape=(1, features))  # Array dei centroidi

    c = np.asarray(dataset.loc[np.random.randint(points * k - 1, size=1)].values.flatten().tolist())  # Restituisce la posizione di un punto scelto a caso dal dataset come vettore; sarà il primo centroide

    dist = np.zeros((1, 2))  # Vettore delle distanze tra i punti del dataset ed i centroidi; le colonne 1 e 2 indicano il numero di riga nel dataset dei punti tra i quali è stata calcolata la distanza

    for i in range(0, len(dataset)):  # Ciclo sulle righe del dataset
        x = np.array(dataset.loc[i, :])  # Riga i-esima del dataset (ovvero punto i-esimo)
        dist = np.append(dist, np.array([[i, distance(x, c)]]), axis=0)
    dist = np.delete(dist, 0, axis=0)  # Cancella la primissima riga della matrice delle distanze, creata inizialmente vuota per avere un array di base cui aggiungere righe
    np.set_printoptions(suppress=True)  # Rimuove la notazione scientifica (per print più puliti)

    # Calcola i k-1 punti più lontani dal primo centroide
    max = np.argsort(dist[:, 1])[-(k-1):]  # Ritorna gli indici nel dataset dei k-1 punti più lontani

    centroids = np.append(centroids, np.array(np.matrix(c)), axis=0)
    v = np.zeros(shape=(1, 2))  # Array temporaneo, per l'aggiunta di ogni punto più lontano alla matrice dei centroidi
    for i in range(len(max)):
        v[0,0] = dataset.loc[max[i], 0]
        v[0,1] = dataset.loc[max[i], 1]
        centroids = np.append(centroids, v, axis=0)
    centroids = np.delete(centroids, 0, axis=0)
    centroids = centroids.round()

    print(centroids)

    # TODO Da aggiustare (l'algoritmo è sbagliato, la distanza massima va calcolata dai punti rimanenti)

    return centroids
