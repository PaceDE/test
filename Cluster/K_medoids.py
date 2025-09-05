import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    'Satisfaction': [2.0, 2, 5.0, 6.0, 4.5, 7.0, 3.0],
    'Education_Level': [1.0, 2, 4.0, 6.5, 5.0, 7.5, 2]
}
df = pd.DataFrame(data)
X = df.values

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def kmedoids(X, k=2, max_iter=100):
    np.random.seed(42)
    medoids = X[np.random.choice(len(X), k, replace=False)]
    
    for _ in range(max_iter):
        labels = []
        for point in X:
            distances = [euclidean_distance(point, medoid) for medoid in medoids]
            labels.append(distances.index(min(distances)))
        labels = np.array(labels)
        
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                total_distances = [sum(euclidean_distance(p, q) for q in cluster_points) for p in cluster_points]
                medoids[i] = cluster_points[np.argmin(total_distances)]
                
    return labels, medoids

labels, medoids = kmedoids(X, k=2)
df['Cluster'] = labels

colors = ['green', 'purple', 'blue', 'orange', 'red']

for i in range(len(medoids)):
    cluster_points = X[labels == i]
    print(f"\nCluster {i}")
    print(pd.DataFrame(cluster_points, columns=df.columns[:-1]))
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i}')
    plt.scatter(medoids[i, 0], medoids[i, 1], color=colors[i], marker='X', s=150)

print("\nName: Dipesh Shrestha \nRoll no: 08\n")
plt.xlabel('Satisfaction')
plt.ylabel('Education Level')
plt.title('K-Medoids Clustering')
plt.legend()
plt.show()
