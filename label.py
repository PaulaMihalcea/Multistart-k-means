import numpy as np

from centroid_distance import centroid_distance


def label(dataset, centroids, k, features):

    dist = centroid_distance(dataset, centroids, k)  # For each point, computes the distance from every centroid

    labels = np.zeros([len(dataset), 2])  # Array containing the cluster number of each point
    index = k - 1
    for i in range(0, len(dist)):
        # Extracts, for each iteration, the subset of distances of a point from all centroids, then finds the minimum
        if (index + (i * k) - (index - 1)) < len(dist):
            point_dist = np.zeros([k, 2])
            for j in range(0, k):
                point_dist[j,0] = index-j
                point_dist[j,1] = dist[index + (i * k) - j, 2]
            labels[i, 0] = i  # Point index (in the dataset)
            labels[i, 1] = point_dist[np.argmin(point_dist[:, 1]), 0]  # Number of the cluster the point belongs to

    dataset[features] = labels[:, 1]  # Now the dataset has a new column containing the cluster to which each point belongs to

    return dataset
