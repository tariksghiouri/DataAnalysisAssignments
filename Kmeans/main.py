import pandas as pd
import numpy as np
import random
from math import sqrt
import matplotlib.pyplot as plt


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


def calc_distance(X1, X2):
    return (sum((X1 - X2) ** 2)) ** 0.5

# matrice
data = np.array([[1, 5],
                 [3, 1],
                 [4, 2],
                 [4, 3],
                 [1, 4],
                 [3, 2],
                 [4, 4],
                 [1, 1],
                 [5, 5],
                 [2, 3]])


init_centroids = random.sample(range(0, len(data)), 3)
print("Initial centroid indexes:", init_centroids)
print("==================================")

centroids = []
for i in init_centroids:
    centroids.append(data[i])

print("Initial centroids:", centroids)
print("==================================")


def findClosestCentroids(ic, X):
    assigned_centroid = []
    for i in X:
        distance = []
        for j in ic:
            distance.append(euclidean_distance(i, j))
        assigned_centroid.append(np.argmin(distance))
    return assigned_centroid


get_centroids = findClosestCentroids(centroids, data)
print("Assigned Centroids:", get_centroids)


def calc_New_Centroids(clusters, X):
    new_centroids = []
    for c in set(clusters):
        current_cluster = X[np.where(np.array(clusters) == c)[0]]
        cluster_mean = np.mean(current_cluster, axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids


def kMeans(data, k, n_iterations):
    init_centroids = random.sample(range(0, len(data)), k)
    centroids = []
    for i in init_centroids:
        centroids.append(data[i])

    for iteration in range(n_iterations):
        get_centroids = findClosestCentroids(centroids, data)

        new_centroids = calc_New_Centroids(get_centroids, data)

        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    return get_centroids, centroids


n_iterations = 10  # number of iterations

assigned_centroids, final_centroids = kMeans(data, 3, n_iterations)

print("Assigned Centroids:", assigned_centroids)
print("Final Centroids:", final_centroids)
