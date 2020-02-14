import numpy as np


def gen_rand_centroids(k, features, center_range):

    c = np.zeros(shape=(1, features))  # Single centroid
    centroids = np.zeros(shape=(1, features))  # Centroids matrix

    for i in range(0, k):
        for j in range(0, features):  # Creates a single centroid
            c[0, j] = int(np.random.randint(center_range, size=1))
        centroids = np.append(centroids, c, axis=0)  # Adds the newly generated centroid to the matrix containing the others

    centroids = np.delete(centroids, 0, axis=0)  # Deletes the first row of the matrix (it was all zeros)
    centroids = centroids.round()  # Rounds the coordinates to integers

    return centroids
