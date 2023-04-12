import pandas as pd
import numpy as np

city_data = pd.read_csv('tetuan_city_power_consumption.csv', skiprows=1, encoding='latin1', usecols=range(1, 8))
subset_size = 2000  # adjust this to your desired subset size (ideal = 3000-4000)
subset_indices = np.random.choice(city_data.index, subset_size, replace=False)
subset = city_data.loc[subset_indices]

def dbscan(df, eps, minPts):

    labels = [0] * len(df)
    C = 0 # id of cluster
    for P in range(0, len(df)):
        if labels[P] != 0:
            continue

        neighbors = threshold(df, P, eps)

        if len(neighbors) < minPts: # if its clusters size is smaller than min, points are noise
            labels[P] = -1
        else:
            C += 1 # grow new cluster from this point if enough neighbors
            growCluster(df, labels, P, neighbors, C, eps, minPts)
    return labels # list indicating cluster membership

# searches through data matrix to find all points that belong to new cluster
def growCluster(df, labels, P, NeighborPts, C, eps, minPts):
    labels[P] = C
    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
            pass # neighbor is noise - ignore
        elif labels[Pn] == 0: # if unvisited
            labels[Pn] = C # add to cluster
            PnNeighborPts = threshold(df, Pn, eps) # check neighbors
            if len(PnNeighborPts) >= minPts: # if it has enough neighbors to be a cluster, add to labels list
                NeighborPts += PnNeighborPts
        i += 1 # goto next neighbor

# gets neighbors within epsilon distance of point
def threshold(df, P, eps):
    neighbors = []
    for neighborPoint in range(0, len(df)):
        if neighborPoint == P:
            continue
        distance = np.linalg.norm(df.iloc[P].values - df.iloc[neighborPoint].values)
        # if euclidean distance between two points is smaller than epsilon, its a neighbor.
        if distance < eps:
            neighbors.append(neighborPoint)
    return neighbors

# NOTE: epsilon needs to be pretty big to get any clusters.
print(dbscan(subset, 3, 5))