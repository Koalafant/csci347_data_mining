import pandas as pd
import numpy as np
import sklearn as sk
import math
import matplotlib.pyplot as plt

city_data = pd.read_csv('tetuan_city_power_consumption.csv')

# e (epsilon): the radius of a neighborhood centered on a given point
# min: minpts - required number of points
def dbscan(df, min, e):

    # numpy array to hold the labels representing noise, border, and core points (all integers). Size of df.
    labels = np.zeros(df.shape[0], dtype=int)

    # function to get euclidean distance between points.
    # def distance(x, y):
    #     d = math.dist(x, y)
    #     return d

    def threshold(df, P, e):
        neighbors = []
        for point in range(df.shape[0]):
            if np.linalg.norm(df[P] - df[point]) < e:
                neighbors.append(point)
        return neighbors

    def new_cluster(df, label, P, neighbors, numClusters, e, min):
        label[P] = numClusters
        S = set(neighbors)
        while S:
            Pn = S.pop()
            if labels[Pn] == -1:
                labels[Pn] = numClusters
            elif labels[Pn] == 0:
                labels[Pn] = numClusters
            PnNeighbors = threshold(df, Pn, e)
            if len(PnNeighbors) >= min:
                S.update(set(PnNeighbors))

    # C = current cluster
    numClusters = 0
    # P will be index of the data point.
    for P in range(df.shape[0]):
        # Only pick unclaimed points
        if not (labels[P] == 0):
            continue
        # neighbors = indexes of all neighboring points
        neighbors = threshold(df, P, e)
        if len(neighbors) < min:
            labels[P] = -1
        else:
            numClusters += 1

