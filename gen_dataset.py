import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plot=True (default: False) permette di visualizzare il grafico del dataset ottenuto
# centroids=True (default: False) permette di ritornare anche la matrice delle coordinate dei centri effettivi dei cluster, così come sono stati definiti nella loro creazione (NON corrispondono a punti effettivamente presenti nel dataset) [for debug purposes only]


def gen_dataset(points=1000, features=2, k=8, center_range=5000, scale_low=0, scale_high=300, plot=False, centers=False):

    samples = []  # Array di cluster (è il samples generato)
    dataset = pd.DataFrame()
    c = np.empty(shape=(1,features))  # Array dei centri dei cluster; for debug purposes only

    # Crea 'clusters' cluster bidimensionali di centro 'center', e li plotta
    for i in range(0, k):
        center = np.empty(shape=(1,features))
        for j in range(0, features):
            center[0, j] = int(np.random.randint(center_range, size=1))
        c = np.append(c, center, axis=0)

        scale = int(np.random.randint(low=scale_low, high=scale_high, size=1))

        samples.append(np.random.normal(loc=center, scale=scale, size=(points, features)))

        dataset = dataset.append(pd.DataFrame(np.asmatrix(samples[i])),ignore_index=True)  # Crea il database di tutti i punti

        # Plotting delle prime due dimensioni dei punti, eseguito per campione
        # plt.scatter(samples[i][:, 0], samples[i][:, 1])  # samples[i][:, 0] è la prima colonna del cluster i; samples[i][:, 1] è la seconda colonna del cluster i

    dataset = dataset.round()

    c = np.delete(c, (0), axis=0)
    c = c.round()

    # Plotting del dataset
    if plot:
        dataset.plot.scatter(x=0,y=1)
        plt.show()

    if centers:
        return dataset, c
    else:
        return dataset
