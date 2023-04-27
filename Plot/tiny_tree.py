import networkx as nx
import matplotlib.pyplot as plt

# Define the tree structure
root = "Root"
level1 = ["Animal", "Others", "Objects"]
edges = [(root, child) for child in level1]

# Create the graph
G = nx.DiGraph()
G.add_edges_from(edges)

# Set node attributes
for n in G:
    G.nodes[n]['layer'] = 0 if n == root else 1

# Plot the graph
pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=15, font_weight="bold", arrowsize=20)
plt.show()
