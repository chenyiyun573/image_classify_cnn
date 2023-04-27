import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

# Define the tree structure
tree_structure = {
    "name": "Root",
    "children": [
        {
            "name": f"Network {i+1}",
            "children": [{"name": f"Network {i+1}-{j+1}"} for j in range(4)]
        } for i in range(5)
    ]
}

# Create the graph
G = nx.DiGraph()

# Add edges to the graph
def add_edges(node, children):
    for child in children:
        G.add_edge(node["name"], child["name"])
        if "children" in child:
            add_edges(child, child["children"])

add_edges(tree_structure, tree_structure["children"])

# Plot the graph using the graphviz_layout
pos = graphviz_layout(G, prog="dot")
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=12, font_weight="bold", arrowsize=20)
plt.show()
