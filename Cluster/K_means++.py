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

def initialize_centroids(X, k):
    np.random.seed(42)
    centroids = [X[np.random.choice(len(X))]]
    for _ in range(1, k):
        distances = np.array([min([euclidean_distance(x, c) for c in centroids]) for x in X])
        probabilities = distances **2/ np.sum(distances**2)
        chosen_index = np.random.choice(len(X), p=probabilities)
        centroids.append(X[chosen_index])
    return np.array(centroids)

def kmeans_pp(X, k=2, max_iter=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iter):
        labels = []
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            labels.append(np.argmin(distances))
        labels = np.array(labels)

        new_centroids = []
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroid = cluster_points.mean(axis=0)
            else:
                new_centroid = centroids[i]
            new_centroids.append(new_centroid)
        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

labels, centroids = kmeans_pp(X, k=2)
df['Cluster'] = labels

colors = ['green', 'purple', 'orange', 'blue', 'red']
for i in range(len(centroids)):
    cluster_points = X[labels == i]
    print(f"\nCluster {i}")
    print(pd.DataFrame(cluster_points, columns=df.columns[:-1]))
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i}')
    plt.scatter(centroids[i, 0], centroids[i, 1], color=colors[i], marker='X', s=150)

print("\nName: Dipesh Shrestha \nRoll no: 08 \n")

plt.xlabel('Satisfaction')
plt.ylabel('Education_Level')
plt.title('K-means++ Clustering')
plt.legend()
plt.show()
