import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add edges to the graph (example)
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 2)])

# Calculate in-degree centrality
in_degrees = dict(G.in_degree())

# Calculate out-degree centrality
out_degrees = dict(G.out_degree())
