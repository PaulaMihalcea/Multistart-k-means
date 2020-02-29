import numpy as np

from gen_rand_centroids import gen_rand_centroids
from label import label


def k_means(dataset, k, features, center_range, scale):

    # K-means initialization

    centroids = gen_rand_centroids(k=k, features=features, center_range=center_range, scale=scale)  # Generates k random centroids

    dataset = label(dataset, centroids, k, features)


    # K-means main loop

    points_in_cluster = np.zeros(k)  # Total number of points in each cluster (after classification)
    features_sum = np.zeros((k, features))  # Partial sum of each feature, calculated for each cluster

    while True:

        prev_centroids = centroids  # Keeps track of the previous centroids, enabling breaking the loop

        for i in range(0, len(dataset)):  # Computes the centroid for each cluster
            points_in_cluster[int(dataset.loc[i, features])] += 1  # Counts the number of points in each cluster
            for j in range(0, features):  # Sums the features of each point, for each cluster
                features_sum[int(dataset.loc[i, features]), j] += dataset.loc[i, j]

        for i in range(0, k):  # For each cluster, calculates the new centroid
            for j in range(0, features):
                centroids[i, j] = features_sum[i, j] / points_in_cluster[i]

        labeled_dataset = label(dataset.drop(columns=[features]), centroids, k, features)  # Re-assigns the points in function of the new centroids

        if (centroids == prev_centroids).all():  # Breaks the loop if centroids are not moving anymore
            break

    return labeled_dataset, centroids
