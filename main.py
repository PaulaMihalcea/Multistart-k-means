from gen_dataset import gen_dataset
from k_means import k_means
from error import error
from plot import plot

##################################################

# Multistart parameters
n = 10  # Number of times multistart should be applied to the dataset

# Dataset parameters
k = 2  # Number of clusters
points = 100  # Number of points per cluster
features = 2  # Number of features of each point

# Normal distribution parameters
center_range = 5000  # Range of coordinates that can be assigned to the clusters' original centers (mean)
scale_low = 500  # Lowest standard deviation value for the normal distribution used to generate the points
scale_high = 1000  # Highest standard deviation value for the normal distribution used to generate the points

# Dataset creation
dataset = gen_dataset(points=points, features=features, k=k, center_range=center_range, scale_low=scale_low, scale_high=scale_high)  # Creates a clusterable set of points

##################################################

# Multistart first iteration
labeled_dataset, centroids = k_means(dataset, k, features, center_range, scale_high)  # K-means algorithm is applied to the data and returns a labeled dataset
e = error(labeled_dataset, centroids)  # For each point, computes its squared distance from the centroid of the cluster the point belongs to

# Multistart subsequent iterations
if n >= 1:
    for i in range(0, n):
        labeled_dataset_new, centroids_new = k_means(dataset, k, features, center_range, scale_high)  # K-means algorithm is applied to the data and returns a labeled dataset
        e_new = error(labeled_dataset_new, centroids_new)  # For each point, computes its squared distance from the centroid of the cluster the point belongs to

        if e_new <= e:  # If the error calculated for the new k-means iteration is smaller, then the old (or first) solution is replaced with the new one
            labeled_dataset = labeled_dataset_new
            centroids = centroids_new
            e = e_new

##################################################

# Plotting (only for datasets with two features)

if features == 2:
    plot(labeled_dataset, centroids, features)
