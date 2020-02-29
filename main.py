import numpy as np
import matplotlib.pyplot as plt

from gen_dataset import gen_dataset
from k_means import k_means
from error import error

##################################################

# Multistart parameters
n = 2  # Number of times k-means should be applied to the dataset
e = np.zeros(n)  # TODO Comment

# Dataset parameters

k = 2  # Number of clusters
points = 10  # Number of points per cluster
features = 2  # Number of features of each point

# Normal distribution parameters
center_range = 5000  # Range of coordinates that can be assigned to the clusters' original centers (mean)
scale_low = 500  # Lowest standard deviation value for the normal distribution used to generate the points
scale_high = 1000  # Highest standard deviation value for the normal distribution used to generate the points

# Dataset creation
dataset = gen_dataset(points=points, features=features, k=k, center_range=center_range, scale_low=scale_low, scale_high=scale_high)  # Creates a clusterable set of points

##################################################

# TODO Procedimento
'''
for i in range(0,n):
    # fai girare k-means (genera centroidi a caso e avvia una ricerca locale da lì)
    # calcola il parametro per valutare la soluzione trovata e salvalo
    # confronta il parametro con quello precedente (dopo la prima iterazione)
    # se è minore di quello precedente, sostituisci quello precedente con quello nuovo, altrimenti lascialo com'è
    pass
'''


# TODO Provare a usare come parametro per decidere se una soluzione è migliore la somma minima delle distanze al quadrato di ogni punto dal proprio centroide
labeled_dataset, centroids = k_means(dataset, k, features, center_range, scale_high)  # K-means algorithm is applied to the data and returns a labeled dataset
error(labeled_dataset, centroids, k, sq=False)  # For each point, computes the squared distance from every centroid

##################################################

# Plotting (only for datasets with two features)

if features == 2:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=labeled_dataset[0], y=labeled_dataset[1], c=labeled_dataset[features], cmap="tab20b", label='Dataset points', s=15)
    ax.scatter(x=centroids[:, 0], y=centroids[:, 1], c='red', marker='x', s=50, label='Centroids')
    plt.show()
