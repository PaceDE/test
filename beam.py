graph = {
    'A': ['B', 'C','D'],
    'B': ['A', 'E'],
    'C': ['A','F', 'E'],
    'D': ['A','F'],
    'E': ['B','C','H'],
    'F': ['C','D','G'],
    'G': ['F','H'],
    'H': ['E','G']
}

heuristic = {
    'A': 40,
    'B': 32,
    'C': 25,
    'D': 35,
    'E': 19,
    'F': 17,
    'G': 0, # Goal
    'H': 10  
}

def beam_search(start, goal, beam_width, max_steps):
    open = [start]
    close = [start]
    parent = {start: None} 
    for step in range(max_steps):
        candidates = []
        for node in open:
            for child in graph[node]:
                if child not in close and child not in parent:
                    candidates.append((child, node))  

        if not candidates:
            break

        
        candidates.sort(key=lambda x: heuristic.get(x[0], 1000))

        open = [node for node, p in candidates[:beam_width]]
        print(f"Step {step+1}: Open= {open}")
        print(f"Step {step+1}: Close = {close}\n")
        for node, p in candidates[:beam_width]:
            parent[node] = p
            if node not in close:
                close.append(node)

        if goal in open:
            print("Goal reached!")
            break

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent.get(node)
    path.reverse()

    return close, path

goal = "G"
close, path = beam_search('A', goal, beam_width=2, max_steps=15)

print("Visited nodes (Close):")
print(close)
print("\nPath from start to goal:")
print(" -> ".join(path))
print("\n Name: Dipesh Shrestha \n Roll no:08 \n")