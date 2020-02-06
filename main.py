import numpy as np
import matplotlib.pyplot as plt

from distance import distance
from gen_dataset import gen_dataset
from gen_centroids import gen_centroids

# INIZIALIZZAZIONE (parametri)

points = 10  # Numero di punti da generare per ogni cluster (righe della matrice)
features = 2  # Numero di dimensioni per ogni punto (colonne della matrice)

clusters = 5
center_range = 5000  # Range di possibili coordinate da assegnare i centri dei cluster
scale_range = 300  # Range della deviazione standard di ogni cluster

# CREAZIONE DATASET

# Crea un dataset di punti suddivisi in cluster utilizzando i parametri specificati in precedenza
# plot=True (default: False) permette di visualizzare il grafico del dataset ottenuto
# centroids=True (default: False) permette di ritornare anche la matrice delle coordinate dei centri effettivi dei cluster, così come sono stati definiti nella loro creazione (NON corrispondono a punti effettivamente presenti nel dataset); for debug purposes only
dataset, centri = gen_dataset(points=points, features=features, clusters=clusters, center_range=center_range, scale_range=scale_range, centroids=True)


# SCELTA CENTROIDI

# TODO Vanno scelti DAL DATASET (devono appartenere all'insieme ammissibile)
# TODO Attenzione: i centroidi attualmente sono generati casualmente e NON coincidono con alcun punto nel dataset (a meno di qualche punto casualmente generato proprio lì)

random_centroids = gen_centroids(clusters, features, center_range)

c = np.empty(shape=(1, features))
centroids = np.empty(shape=(1, features))  # Array dei centri dei cluster; for debug purposes only

for i in range(0, clusters):
    #print(np.random.randint(points*clusters-1, size=1))
    c = dataset.loc[np.random.randint(points*clusters-1, size=1)]  # Restituisce la posizione di un punto scelto a caso dal dataset
    c = c.values
    #print(c)
    centroids = np.append(centroids, c, axis=0)  # Aggiunta del centroide alla matrice che li contiene tutti
centroids = np.delete(centroids, 0, axis=0)
centroids = centroids.round()

print(dataset)
print()
print(centroids)
print()
print(random_centroids)



# CALCOLO DISTANZE PUNTI-CENTROIDI

# Calcolo distanze tra punti e centroidi
dist = np.zeros((1, 3))  # Vettore delle distanze tra i punti del dataset ed i centroidi; le colonne 1 e 2 indicano il numero di riga nel dataset dei punti tra i quali è stata calcolata la distanza

for i in range(0, len(dataset)):  # Ciclo sulle righe del dataset
    for j in range(0, clusters):  # Ciclo sui centroidi (uguali in numero ai cluster)  # TODO Tecnicamente non sappiamo quanti cluster ci sono, quindi andrà aggiustato anche questo numero
        x = np.array(dataset.loc[i, :])  # Riga i-esima del dataset (ovvero punto i-esimo)
        y = centroids[j, :]  # Centroide i-esimo
        dist = np.append(dist, np.array([[i, j, distance(x, y)]]), axis=0)
dist = np.delete(dist, 0, axis=0)  # Cancella la primissima riga della matrice delle distanze, creata inizialmente vuota per avere un array di base cui aggiungere righe
np.set_printoptions(suppress=True)  # Rimuove la notazione scientifica (per print più puliti)


# CLASSIFICAZIONE (individuazione dei cluster)

labels = np.zeros([len(dataset), 2])  # Vettore contenente il cluster di appartenenza di ogni punto
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

# PLOTTING

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x=dataset[0], y=dataset[1], c=dataset[2], cmap="tab20b", label='Dataset points')
ax.scatter(x=centroids[:, 0], y=centroids[:, 1], c='red', marker='^', label='Centroids')
#plt.legend(loc='upper left')

plt.show()

# TODOs

# TODO Unificare la lingua in commenti e nomi variabili/funzioni (inglese o italiano?)
# TODO Implementare scelta del numero di cluster (che per il momento è nota)
# TODO Dare ai centroidi lo stesso colore dei cluster cui si riferiscono (al momento i centroidi sono tutti rossi)
