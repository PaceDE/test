import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


eps = 3.5
min_pts = 3


data = {
    "X": [5, 8, 3, 4, 3, 6, 6, 5],
    "Y": [7, 4, 3, 4, 7, 7, 1, 5]
}

df = pd.DataFrame(data)
data = df.values
n = len(data)


core_points = []
neighbors_list = []

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

for i in range(n):
    neighbors = []
    for j in range(n):
        if euclidean(data[i], data[j]) <= eps:
            neighbors.append(j)
    neighbors_list.append(neighbors)
    if len(neighbors) >= min_pts:
        core_points.append(i)


point_type = []
for i in range(n):
    if i in core_points:
        point_type.append("Core")
    else:
       
        is_border = any(i in neighbors_list[core] for core in core_points)
        if is_border:
            point_type.append("Border")
        else:
            point_type.append("Noise")


print(f"{'Point':<6} {'Core/Noise':<12} {'Border/Noise'}")

for i in range(n):
    name = f"P{i+1}"
    ptype = point_type[i]
    if ptype == "Core":
        core_noise = "Core"
        boundary_noise = ""
    elif ptype == "Border":
        core_noise = "Noise"
        boundary_noise = "Boundary"
    else:
        core_noise = "Noise"
        boundary_noise = "Noise"
    print(f"{name:<6} {core_noise:<12} {boundary_noise}")

color_map = {"Core": "green", "Border": "orange", "Noise": "gray"}
plt.figure(figsize=(8, 5))
for i in range(n):
    plt.scatter(data[i][0], data[i][1], color=color_map[point_type[i]], edgecolors='black', s=70)
    plt.text(data[i][0]+0.1, data[i][1]+0.1, f"P{i+1}", fontsize=9)

plt.title("Simple DBSCAN Point Types")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
