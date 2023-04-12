import pandas as pd
import numpy as np

city_data = pd.read_csv('tetuan_city_power_consumption.csv', skiprows=1, encoding='latin1', usecols=range(1, 8))
subset_size = 2000  # adjust this to your desired subset size (ideal = 3000-4000)
subset_indices = np.random.choice(city_data.index, subset_size, replace=False)
subset = city_data.loc[subset_indices]

def dbscan(D, eps, MinPts):

    labels = [0] * len(D)
    C = 0 # id of cluster
    for P in range(0, len(D)):
        if labels[P] != 0:
            continue

        neighbors = region_query(D, P, eps)

        if len(neighbors) < MinPts: # if its clusters size is smaller tha min, points are noise
            labels[P] = -1
        else:
            C += 1 # grow new cluster from this point is enough neighbors
            grow_cluster(D, labels, P, neighbors, C, eps, MinPts)
    return labels # list indicating cluster membership

# searches through data matrix to find all points that belong to new cluster
def grow_cluster(D, labels, P, NeighborPts, C, eps, MinPts):
    labels[P] = C
    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
            pass # neighbor is noise - ignore
        elif labels[Pn] == 0: # if unvisited
            labels[Pn] = C # add to cluster
            PnNeighborPts = region_query(D, Pn, eps) # check neighbors
            if len(PnNeighborPts) >= MinPts: # if it has enough neighbors to be a cluster, add to labels list
                NeighborPts += PnNeighborPts
        i += 1 # goto next neighbor

# gets neighbors within epsilon distance of point
def region_query(D, P, eps):
    neighbors = []
    for neighborPoint in range(0, len(D)):
        if neighborPoint == P:
            continue
        distance = np.linalg.norm(D.iloc[P].values - D.iloc[neighborPoint].values)
        # if euclidean distance between two points is smaller than epsilon, its a neighbor.
        if distance < eps:
            neighbors.append(neighborPoint)
    return neighbors

# NOTE: epsilon needs to be pretty big to get any clusters.
print(dbscan(subset, 3, 5))