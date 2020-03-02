import numpy as np

from centroids_distance import centroids_distance


def label(dataset, centroids, k, features):

    dist = centroids_distance(dataset, centroids, k)  # For each point, computes the distance from every centroid

    labels = np.zeros([len(dataset), 2])  # Array containing the cluster number of each point
    index = k - 1

    for i in range(0, len(dist)):
        if (index + (i * k) - (index - 1)) < len(dist):  # Extracts, for each iteration, the subset of distances of a point from all centroids, then finds the minimum
            point_dist = np.zeros([k, 2])

            for j in range(0, k):
                point_dist[j, 0] = index - j
                point_dist[j, 1] = dist[index + (i * k) - j, 2]

            labels[i, 0] = i  # Adds the points index to the 'labels' array first column
            labels[i, 1] = point_dist[np.argmin(point_dist[:, 1]), 0]  # Adds the number of the cluster the point belongs to ('labels' second column)

    labeled_dataset = dataset.copy(deep=True)  # Deep copies the original dataset to avoid modifying it with the labels
    labeled_dataset[features] = labels[:, 1]  # Now the dataset has a new column containing the cluster to which each point belongs to

    return labeled_dataset
