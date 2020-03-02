import numpy as np

from distance import distance


def error(labeled_dataset, centroids):

    distances_sum = 0

    for i in range(0, len(labeled_dataset)):  # Loop on the dataset rows (or points)
        x = np.array(labeled_dataset.loc[i, :(len(labeled_dataset.columns)-2)])  # i-th row of the dataset (also i-th point)
        c = centroids[int(labeled_dataset.loc[i, len(labeled_dataset.columns)-1]), :]  # Centroid of x's cluster

        distances_sum += distance(x, c, sq=True)  # Sums the squared distance of x from its centroid, c, for each point x in the dataset

    np.set_printoptions(suppress=True)  # Removes scientific notation (better prints)

    return distances_sum
