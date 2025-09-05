import numpy as np
import matplotlib.pyplot as plt

# 0=unvisited, -1=noise, cluster_id>=1

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def neighbor(data, point_idx, eps):
    neighbors = []
    for i in range(len(data)):
        if euclidean_distance(data[point_idx], data[i]) <= eps:
            neighbors.append(i)
    return neighbors

def expand_cluster(data, labels, point_idx, neighbors, cluster_id, eps, min_pts):
    """Expand the cluster using density connectivity"""
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        n_idx = neighbors[i]
        if labels[n_idx] == -1:  
            labels[n_idx] = cluster_id
        elif labels[n_idx] == 0: 
            labels[n_idx] = cluster_id
            new_neighbors = neighbor(data, n_idx, eps)
            if len(new_neighbors) >= min_pts:
                neighbors += new_neighbors
        i += 1

def dbscan(data, eps, min_pts):
    labels = [0] * len(data)  
    cluster_id = 0
    for i in range(len(data)):
        if labels[i] != 0:
            continue
        neighbors = neighbor(data, i, eps)
        if len(neighbors) < min_pts:
            labels[i] = -1  
        else:
            cluster_id += 1
            expand_cluster(data, labels, i, neighbors, cluster_id, eps, min_pts)
    return labels


data = np.array([
    [1, 3],[8,8], [2, 4], [3, 3],
     [10, 8],
    [15, 20]  
])

eps = 2
min_pts = 2
labels = dbscan(data, eps, min_pts)

plt.figure(figsize=(6, 5))
colors = ['r', 'g', 'b', 'c', 'm', 'y']

for idx, point in enumerate(data):
    if labels[idx] == -1:
        plt.scatter(point[0], point[1], c='k', s=100, edgecolors='k', label="Noise" if idx == 0 else "")
        print(f"Noise: {point[0]},{point[1]}")
    else:
        plt.scatter(point[0], point[1], c=colors[labels[idx] % len(colors)], s=100, edgecolors='k', label=f"Cluster {labels[idx]}" if f"Cluster {labels[idx]}" not in plt.gca().get_legend_handles_labels()[1] else "")
        print(f"Cluster {labels[idx]}: {point[0]},{point[1]}")
    plt.text(point[0]+0.2, point[1]+0.2, str(idx))


print("\nName: Dipesh Shrestha \nRoll no:08 \n")
plt.title("DBSCAN")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

