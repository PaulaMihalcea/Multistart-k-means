import matplotlib.pyplot as plt


def plot(labeled_dataset, centroids, features):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=labeled_dataset[0], y=labeled_dataset[1], c=labeled_dataset[features], cmap="tab20b", label='Dataset points', s=15)
    ax.scatter(x=centroids[:, 0], y=centroids[:, 1], c='red', marker='x', s=50, label='Centroids')
    plt.show()
