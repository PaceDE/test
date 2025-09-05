import pandas as pd
from collections import Counter
import numpy as np

data = pd.read_excel('ID3.xlsx')
data = data.to_dict(orient='records')  

def entropy(data):
    labels = [row['PlayTennis'] for row in data]
    total = len(labels)
    counts = Counter(labels)
    return -sum((count/total) * np.log2(count/total) for count in counts.values())

def info_gain(data, feature):
    total_entropy = entropy(data)
    values = set(row[feature] for row in data)
    weighted_entropy = 0
    for val in values:
        subset = [row for row in data if row[feature] == val]
        weighted_entropy += len(subset)/len(data) * entropy(subset)
    return total_entropy - weighted_entropy

def ID3(data, features):
    labels = [row['PlayTennis'] for row in data]
    if labels.count(labels[0]) == len(labels):
        return labels[0]  

    if not features:
        return Counter(labels).most_common(1)[0][0]  

    # Choose best feature
    gains = [info_gain(data, f) for f in features]
    best_feature = features[gains.index(max(gains))]

    tree = {best_feature: {}}
    values = set(row[best_feature] for row in data)

    for val in values:
        subset = [row for row in data if row[best_feature] == val]
        sub_features = [f for f in features if f != best_feature]
        tree[best_feature][val] = ID3(subset, sub_features)

    return tree

def print_tree(tree, indent=""):
    for key, value in tree.items():
        for val, sub in value.items():
            print(f"{indent}{key} = {val} ->", end=" ")
            if isinstance(sub, dict):
                print()
                print_tree(sub, indent + "    ")
            else:
                print(sub)

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    root = next(iter(tree))
    value = sample.get(root)
    if value not in tree[root]:
        
        sub_values = tree[root].values()
        labels = [v for v in sub_values if isinstance(v, str)]
        if labels:
            return Counter(labels).most_common(1)[0][0]
        else:
            value = next(iter(tree[root]))
    return predict(tree[root][value], sample)

features = list(data[0].keys())
features.remove('PlayTennis')

# Build tree
tree = ID3(data, features)

# Print human-readable tree
print("Decision tree:")
print_tree(tree)

sample = {'Outlook':'Sunny', 'Temperature':'Hot', 'Humidity':'High', 'Windy':'False'}
prediction = predict(tree, sample)
print("\nPrediction for sample:", prediction)

print("\n Name: Dipesh Shrestha \n Roll no:08 \n")
