import numpy as np

from distance import distance

def centroid_distance(dataset, centroids, k):

    dist = np.zeros((1, 3))  # Vettore delle distanze tra i punti del dataset ed i centroidi; le colonne 1 e 2 indicano il numero di riga nel dataset dei punti tra i quali è stata calcolata la distanza

    for i in range(0, len(dataset)):  # Ciclo sulle righe del dataset
        for j in range(0, k):  # Ciclo sui centroidi (uguali in numero ai cluster)  # TODO Tecnicamente non sappiamo quanti cluster ci sono, quindi andrà aggiustato anche questo numero
            x = np.array(dataset.loc[i, :])  # Riga i-esima del dataset (ovvero punto i-esimo)
            y = centroids[j, :]  # Centroide i-esimo
            dist = np.append(dist, np.array([[i, j, distance(x, y)]]), axis=0)
    dist = np.delete(dist, 0, axis=0)  # Cancella la primissima riga della matrice delle distanze, creata inizialmente vuota per avere un array di base cui aggiungere righe
    np.set_printoptions(suppress=True)  # Rimuove la notazione scientifica (per print più puliti)

    print(dist)   # TODO

    return dist
