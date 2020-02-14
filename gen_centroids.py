import numpy as np

def gen_centroids(dataset, points, k, features):

    centroids = np.empty(shape=(1, features))  # Array dei centroidi

    for i in range(0, k):
        c = dataset.loc[np.random.randint(points * k - 1, size=1)]  # Restituisce la posizione di un punto scelto a caso dal dataset
        c = c.values
        centroids = np.append(centroids, c, axis=0)  # Aggiunta del centroide alla matrice che li contiene tutti
    centroids = np.delete(centroids, 0, axis=0)
    centroids = centroids.round()

    return centroids
