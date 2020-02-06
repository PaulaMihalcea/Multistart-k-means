import numpy as np
import matplotlib.pyplot as plt

from distance import distance
from create_clusters import create_clusters

points = 100  # Numero di punti da generare per ogni cluster (righe della matrice)
features = 2  # Numero di dimensioni per ogni punto (colonne della matrice)

clusters = 5
center_range = 5000  # Range di possibili coordinate da assegnare i centri dei cluster
scale_range = 100  # Range della deviazione standard di ogni cluster

# Crea un dataset di punti suddivisi in cluster utilizzando i parametri specificati in precedenza
# plot=True (default: False) permette di visualizzare il grafico del dataset ottenuto
# centroids=True (default: False) permette di ritornare anche la matrice delle coordinate dei centri effettivi dei cluster, così come sono stati definiti nella loro creazione (NON corrispondono a punti effettivamente presenti nel dataset); for debug purposes only
dataset, centri = create_clusters(points=points, features=features, clusters=clusters, center_range=center_range, scale_range=scale_range, centroids=True)

# TODO Va scelto in modo random
# TODO Attenzione: i centroidi attualmente NON coincidono con alcun punto nel dataset (a meno di qualche punto casualmente generato proprio lì)

# Calcolo distanze tra punti e centroidi
dist = np.zeros((1, 3))  # Vettore delle distanze tra i punti del dataset ed i centroidi; le colonne 1 e 2 indicano il numero di riga nel dataset dei punti tra i quali è stata calcolata la distanza

for i in range(0, len(dataset)):  # Ciclo sulle righe del dataset
    for j in range(0, clusters):  # Ciclo sui centroidi (uguali in numero ai cluster)  # TODO Tecnicamente non sappiamo quanti cluster ci sono, quindi andrà aggiustato anche questo numero
        x = np.array(dataset.loc[i, :])  # Riga i-esima del dataset (ovvero punto i-esimo)
        y = centri[j, :]  # Centroide i-esimo
        dist = np.append(dist, np.array([[i, j, distance(x, y)]]), axis=0)
dist = np.delete(dist, 0, axis=0)  # Cancella la primissima riga della matrice delle distanze, creata inizialmente vuota per avere un array di base cui aggiungere righe
np.set_printoptions(suppress=True)


labels = np.zeros([len(dataset), 2])  # Vettore contenente

index = clusters - 1
for k in range(0, len(dist)):
    # Estrae, ad ogni ciclo, il sottoinsieme delle distanze di un punto da tutti i centroidi, per ogni punto, e trova quella minima (insieme al numero del cluster a cui si riferisce)
    if (index + (k * clusters)-(index - 1)) < len(dist):
        point_dist = np.zeros([clusters, 2])
        for j in range(0, clusters):
            point_dist[j,0] = index-j
            point_dist[j,1] = dist[index + (k * clusters) - j, 2]
        labels[k, 0] = k  # Numero del punto
        labels[k, 1] = point_dist[np.argmin(point_dist[:,1]),0]  # Numero del cluster a cui appartiene il punto

dataset[2] = labels[:,1]  # Adesso il dataset contiene una terza colonna indicante il cluster a cui appartiene ogni punto

p = dataset.plot.scatter(x=0,y=1,c=dataset[2],cmap="tab20b")
plt.show()
