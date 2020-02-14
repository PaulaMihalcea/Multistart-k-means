import numpy as np


def gen_rand_centroids(k, features, center_range):

    c = np.empty(shape=(1, features))
    centroids = np.empty(shape=(1, features))  # Array dei centri dei cluster; for debug purposes only
    for i in range(0, k):
        for j in range(0, features):  # Generazione del singolo centroide
            c[0, j] = int(np.random.randint(center_range, size=1))
        centroids = np.append(centroids, c, axis=0)  # Aggiunta del centroide alla matrice che li contiene tutti
    centroids = np.delete(centroids, 0, axis=0)
    centroids = centroids.round()

    return centroids
