import matplotlib.pyplot as plt

from gen_dataset import gen_dataset
from k_means import k_means

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

# K-means

dataset, centroids = k_means(dataset, k, features, center_range, scale_high)

##################################################

# Plotting (only for datasets with two features)

if features == 2:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=dataset[0], y=dataset[1], c=dataset[features], cmap="tab20b", label='Dataset points', s=15)
    ax.scatter(x=centroids[:, 0], y=centroids[:, 1], c='red', marker='x', s=50, label='Centroids')
    plt.show()
