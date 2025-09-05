import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    'Satisfaction': [2.0, 3.5, 5.0, 6.0, 4.5, 7.0, 3.0],
    'Education_Level': [1.0, 2.5, 4.0, 6.5, 5.0, 7.5, 3.5]
}
df = pd.DataFrame(data)
X = df.values

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def kmeans(X, k=2, max_iter=100):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_iter):
        labels = []
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            labels.append(distances.index(min(distances)))
        labels = np.array(labels)
        new_centroids = []
        for i in range(k):
            points_in_cluster = X[labels == i]
            if len(points_in_cluster) > 0:
                new_centroid = points_in_cluster.mean(axis=0)
            else:
                new_centroid = centroids[i]
            new_centroids.append(new_centroid)
        new_centroids = np.array(new_centroids)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

labels, centroids = kmeans(X, k=2)
df['Cluster'] = labels

colors = ['green', 'purple','blue',  'orange','red', ]


for i in range(len(centroids)):
    cluster_points = X[labels == i]
    print(f"\nCluster {i}")
    print(pd.DataFrame(cluster_points, columns=df.columns[:-1]))
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i}')
    plt.scatter(centroids[i, 0], centroids[i, 1], color=colors[i], marker='X', s=150)

print("\nName: Dipesh Shrestha \nRoll no:08 \n")
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.title('K-means Clustering')
plt.legend()
plt.show()
