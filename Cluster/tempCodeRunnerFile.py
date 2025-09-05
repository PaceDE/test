import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def split_cluster(points, data):
    if len(points) <= 1:
        return [points]

    max_dist = -1
    point_a, point_b = points[0], points[0]
    for i in points:
        for j in points:
            dist = euclidean(data[i], data[j])
            if dist > max_dist:
                max_dist = dist
                point_a, point_b = i, j

    cluster1, cluster2 = [], []
    for idx in points:
        if euclidean(data[idx], data[point_a]) < euclidean(data[idx], data[point_b]):
            cluster1.append(idx)
        else:
            cluster2.append(idx)

    return [cluster1, cluster2]

def divisive_clustering(data):
    n = len(data)
    clusters = {0: list(range(n))}  # key=cluster_id, value=list of points
    next_cluster_id = 1
    linkage = []

    # Keep track of which clusters have been split
    to_split = [0]

    while to_split:
        current = to_split.pop(0)
        points = clusters[current]

        if len(points) <= 1:
            continue

        # Split cluster
        c1, c2 = split_cluster(points, data)

        # Assign new cluster IDs
        id1 = next_cluster_id
        next_cluster_id += 1
        id2 = next_cluster_id
        next_cluster_id += 1

        clusters[id1] = c1
        clusters[id2] = c2

        
        del clusters[current]

        # Distance between two child clusters (minimum pairwise distance)
        dist = min(euclidean(data[i], data[j]) for i in c1 for j in c2)

        linkage.append([id1, id2, dist, len(c1) + len(c2)])

        
        if len(c1) > 1:
            to_split.append(id1)
        if len(c2) > 1:
            to_split.append(id2)

    
    linkage = np.array(linkage)
    
    linkage = linkage[linkage[:, 2].argsort()]

    linkage[:, 0] += n - 1
    linkage[:, 1] += n - 1

    return linkage

# Sample data
data = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [5, 7],
    [3, 5],
    [8, 7]
])

linkage = divisive_clustering(data)

plt.figure(figsize=(8, 5))
dendrogram(linkage, labels=np.arange(len(data)))
plt.title("Divisive Hierarchical Clustering Dendrogram")
plt.xlabel("Data Point Index")
plt.ylabel("Distance")
plt.grid(True)
plt.show()
