import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gen_dataset(points=1000, features=2, k=8, center_range=5000, scale_low=500, scale_high=1000, plot=False):

    samples = []  # Array of clusters (with their points)
    dataset = pd.DataFrame()  # Pandas dataframe; will be the final dataset

    for i in range(0, k):  # Creates k centers for the clusters
        center = np.empty(features)
        for j in range(0, features):
            center[j] = int(np.random.randint(center_range, size=1))

        samples.append(np.random.normal(loc=center, scale=int(np.random.randint(low=scale_low, high=scale_high, size=1)), size=(points, features)))  # Appends a newly generated point to its cluster

        dataset = dataset.append(pd.DataFrame(np.asmatrix(samples[i])), ignore_index=True)  # Appends the newly generated cluster to the dataset

    dataset = dataset.round()  # Rounds the coordinates to integers

    # Plotting
    if plot and features == 2:  # plot = True (default: False) shows the plot of the generated dataset (only for datasets with two features)
        dataset.plot.scatter(x=0, y=1)
        plt.show()

    return dataset
