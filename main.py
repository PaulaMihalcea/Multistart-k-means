import numpy as np
import matplotlib.pyplot as plt

from gen_dataset import gen_dataset
from gen_dataset_centroids import gen_dataset_centroids
from centroid_distance import centroid_distance

# INIZIALIZZAZIONE (parametri)
points = 10  # Numero di punti da generare per ogni cluster (righe della matrice)
features = 5  # Numero di dimensioni per ogni punto (colonne della matrice)
k = 3
center_range = 5000  # Range di possibili coordinate da assegnare i centri dei cluster
scale_range = 300  # Range della deviazione standard di ogni cluster

# CREAZIONE DATASET
# Crea un dataset di punti suddivisi in cluster utilizzando i parametri specificati in precedenza
# plot=True (default: False) permette di visualizzare il grafico del dataset ottenuto
# centroids=True (default: False) permette di ritornare anche la matrice delle coordinate dei centri effettivi dei cluster, così come sono stati definiti nella loro creazione (NON corrispondono a punti effettivamente presenti nel dataset) [for debug purposes only]
dataset, centri = gen_dataset(points=points, features=features, k=k, center_range=center_range, scale_range=scale_range, centroids=True)


# SCELTA CENTROIDI
centroids = gen_dataset_centroids(dataset, points, k, features)

# CALCOLO DISTANZE PUNTI-CENTROIDI
dist = centroid_distance(dataset, centroids, k)  # Calcolo distanze tra punti e centroidi


# CLASSIFICAZIONE INIZIALE (individuazione dei cluster)

labels = np.zeros([len(dataset), 2])  # Vettore contenente il cluster di appartenenza di ogni punto
index = k - 1
for i in range(0, len(dist)):
    # Estrae, ad ogni ciclo, il sottoinsieme delle distanze di un punto da tutti i centroidi, per ogni punto, e trova quella minima (insieme al numero del cluster a cui si riferisce)
    if (index + (i * k) - (index - 1)) < len(dist):
        point_dist = np.zeros([k, 2])
        for j in range(0, k):
            point_dist[j,0] = index-j
            point_dist[j,1] = dist[index + (i * k) - j, 2]
        labels[i, 0] = i  # Numero del punto
        labels[i, 1] = point_dist[np.argmin(point_dist[:, 1]), 0]  # Numero del cluster a cui appartiene il punto

dataset[features-1] = labels[:, 1]  # Adesso il dataset contiene una terza colonna indicante il cluster a cui appartiene ogni punto


###########################

# LOOP PRINCIPALE
x = 0  # TODO La regola di arresto va riscritta ('finché i centroidi non vengono più riassegnati')
punti_nel_cluster = np.zeros(k)  # Numero di punti presenti in ogni cluster
somma_features = np.zeros(k)  # Somma parziale delle features per ogni cluster

print(dataset)
#print(dataset.loc[0,2])

while x < 10:
    for i in range(0, len(dataset)):  # Calcola il centroide per ogni cluster
        punti_nel_cluster[int(dataset.loc[i, 4])] += 1
        for j in range(0, features):
            #somma_features[int(dataset.loc[i, j])] += dataset.loc[i, 2]
            pass

    x += 1
    pass


###########################

# PLOTTING
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x=dataset[0], y=dataset[1], c=dataset[features-1], cmap="tab20b", label='Dataset points', s=15)
ax.scatter(x=centroids[:, 0], y=centroids[:, 1], c='red', marker='x', s=50, label='Centroids')
#plt.legend(loc='upper left')

plt.show()  # TODO


# TODOs

# TODO Unificare la lingua in commenti e nomi variabili/funzioni (inglese o italiano?)
# TODO Dare ai centroidi lo stesso colore dei cluster cui si riferiscono (al momento i centroidi sono tutti rossi)
# TODO Rendere i punti più sparsi/meno raggruppati
