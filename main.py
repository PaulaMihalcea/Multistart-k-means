import numpy as np
import matplotlib.pyplot as plt

from gen_dataset import gen_dataset
from gen_rand_centroids import gen_rand_centroids
from label import label

##################################################

# Dataset parameters

k = 2  # Number of clusters
points = 1000  # Number of points per cluster
features = 2  # Number of features of each point

# Normal distribution parameters
center_range = 5000  # Range of coordinates that can be assigned to the clusters' original centers (mean)
scale_low = 500  # Lowest standard deviation value for the normal distribution used to generate the points
scale_high = 1000  # Highest standard deviation value for the normal distribution used to generate the points

# Dataset creation
dataset = gen_dataset(points=points, features=features, k=k, center_range=center_range, scale_low=scale_low, scale_high=scale_high)  # Creates a clusterable set of points

##################################################

# K-means initialization

centroids = gen_rand_centroids(k=k, features=features, center_range=center_range, scale=scale_high)  # Generates k random centroids

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

    dataset = label(dataset.drop(columns=[features]), centroids, k, features)  # Re-assigns the points in function of the new centroids

    if (centroids == prev_centroids).all():  # Breaks the loop if centroids are not moving anymore
        break

##################################################

# Plotting (only for datasets with two features)

if features == 2:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=dataset[0], y=dataset[1], c=dataset[features], cmap="tab20b", label='Dataset points', s=15)
    ax.scatter(x=centroids[:, 0], y=centroids[:, 1], c='red', marker='x', s=50, label='Centroids')
    plt.show()
