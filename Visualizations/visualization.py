import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_diagram_from_matrix(m):
    l = max(m.shape[0], m.shape[1])
    node_positions = {}
    for (i,j), w in np.ndenumerate(m):
        offset = 0.1*j
        node_positions[f"i_{i}"] = (0,l - i)
        node_positions[f"j_{j}"] = (1,l - j)


    # Create a directed graph object
    G = nx.DiGraph()
    G.add_nodes_from(node_positions.keys())

    for (i,j), w in np.ndenumerate(m):
        if w != 0:
            G.add_edge(f"i_{i}", f"j_{j}", weight=w)
            print(f"{i}, {j} -> {w}")

    # Create a dictionary for edge labels (weight)
    edge_labels = {(u, v): G.edges[u, v]['weight'] for u, v in G.edges()}

    # Draw the graph with positions, labels, and arrows for directed edges
    nx.draw(G, node_positions, with_labels=True, arrows=True, node_size=600)
    nx.draw_networkx_edge_labels(G, node_positions, edge_labels=edge_labels)
    plt.show()
    print('************************************')
