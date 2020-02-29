import numpy as np

from distance import distance


def error(dataset, centroids, k):  # TODO Rename function

    dist = np.zeros((1, 3))  # Array of distances of each point from each centroid; columns 0 and 1 refer to the index in the dataset of the points that have been used for the distance stored in column 2

    for i in range(0, len(dataset)):  # Loop on the dataset rows (or points)
        for j in range(0, k):  # Loop on centroids (which are equal in number to k)
            x = np.array(dataset.loc[i, :])  # i-th row of the dataset (also i-th point)
            y = centroids[j, :]  # i-th centroid
            dist = np.append(dist, np.array([[i, j, distance(x, y)]]), axis=0)

    dist = np.delete(dist, 0, axis=0)  # Removes the first row of tha matrix (all zeros)
    np.set_printoptions(suppress=True)  # Removes scientific notation (better prints)

    return dist
