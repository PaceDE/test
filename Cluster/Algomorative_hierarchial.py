import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def agglomerative_clustering(data):
    n = len(data)
    clusters = [[i] for i in range(n)]
    cluster_map = {i: [i] for i in range(n)}
    
    # Initialize cluster names: P0, P1, P2, ...
    cluster_names = {i: f"P{i}" for i in range(n)}
    
    next_cluster_id = n
    linkage_matrix = []
    merge_steps = []
    print("Pairs:")

    while len(clusters) > 1:
        min_dist = float('inf')
        pair = None

        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1 = cluster_map[clusters[i][0]]
                c2 = cluster_map[clusters[j][0]]
                dist = min(euclidean(data[p1], data[p2]) for p1 in c1 for p2 in c2)

                if dist < min_dist:
                    min_dist = dist
                    pair = (i, j)

        i, j = pair
        a, b = clusters[i][0], clusters[j][0]
        new_points = cluster_map[a] + cluster_map[b]
        linkage_matrix.append([a, b, min_dist, len(new_points)])

        
        name_a = cluster_names[a]
        name_b = cluster_names[b]
        new_name = f"({name_a}, {name_b})"
        merge_steps.append(new_name)
        print(new_name)

        
        cluster_names[next_cluster_id] = new_name
        cluster_map[next_cluster_id] = new_points
        clusters = [clusters[x] for x in range(len(clusters)) if x not in [i, j]]
        clusters.append([next_cluster_id])
        next_cluster_id += 1

    return np.array(linkage_matrix), merge_steps


data = np.array([[18], [22], [25], [27], [42], [43]])


linkage, merges = agglomerative_clustering(data)

print("\nName: Dipesh Shrestha \nRoll no:08 \n")
plt.figure(figsize=(8, 5))
dendrogram(linkage, labels=np.arange(len(data)))
plt.title("Agglomerative Clustering Dendrogram (1D Points)")
plt.xlabel("Data Point Index")
plt.ylabel("Distance")
plt.grid(True)
plt.show()
