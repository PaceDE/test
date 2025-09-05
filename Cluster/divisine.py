import numpy as np
import matplotlib.pyplot as plt

# Sample data
data = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [5, 7],
    [3, 5],
    [8, 7]
])

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def split_cluster(points, data):
    """Split a cluster into two using farthest points"""
    if len(points) <= 1:
        return [points]

    # Find two farthest points
    max_dist = -1
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

    return cluster1, cluster2

def plot_clusters(clusters, step_num, data):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    plt.figure(figsize=(6, 4))
    for cluster_idx, cluster in enumerate(clusters):
        cluster_points = data[cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors[cluster_idx % len(colors)], s=100)
        for point in cluster:
            plt.text(data[point, 0]+0.1, data[point, 1]+0.1, str(point))
    plt.title(f"Divisive Clustering - Step {step_num}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def divisive_clustering(data):
    clusters = [list(range(len(data)))]  # start with all points in one cluster
    step_num = 0

    # Initial plot
    print(f"Step {step_num}: {clusters}")
    plot_clusters(clusters, step_num, data)

    while True:
        split_done = False
        new_clusters = []
        for cluster in clusters:
            if len(cluster) > 1:
                c1, c2 = split_cluster(cluster, data)
                new_clusters.extend([c1, c2])
                split_done = True
            else:
                new_clusters.append(cluster)

        if not split_done:
            break

        clusters = new_clusters
        step_num += 1
        print(f"Step {step_num}: {clusters}")
        plot_clusters(clusters, step_num, data)

    return clusters

# Run divisive clustering
final_clusters = divisive_clustering(data)
print("\n Name: Dipesh Shrestha \n Roll no:08 \n")