# --- 1. Build adjacency matrix and run spectral clustering ---
A = nx.to_numpy_array(Graph)  # adjacency matrix ordered by G.nodes()

n_clusters = 18  # tune based on laplacian matrix. 4 and 15
sc = SpectralClustering(
    n_clusters=n_clusters,
    affinity='precomputed',
    assign_labels='kmeans',
    random_state=42
)
labels = sc.fit_predict(A)

# Map node -> cluster label
node_list = list(Graph.nodes())
cluster_map = {node: labels[i] for i, node in enumerate(node_list)}

# --- 2. Assign colors per cluster ---
cmap = cm.get_cmap('tab20', n_clusters)  # 'tab20' has 20 distinct colors
node_colors = [cmap(cluster_map[node]) for node in Graph.nodes()]

# --- 3. Draw ---
plt.figure(figsize=(14, 14))
pos = nx.spring_layout(Graph, k=0.15, iterations=20, weight=None, seed=42)

# Draw edges first (behind nodes)

weights = list(nx.get_edge_attributes(Graph, 'weight').values())
if weights:
    min_weight = min(weights)
    print(min_weight)
    max_weight = max(weights)
    print(max_weight)
    normalized_weights = [
        1 + (x - min_weight) / (max_weight - min_weight) * 4 for x in weights
    ] # normalize to go between 1 and 5
else:
    print("Graph has no edge weights.")
nx.draw_networkx_edges(Graph, pos, width=normalized_weights, alpha=0.4, edge_color='gray')

# Draw nodes colored by cluster
node_weights = [Graph.nodes[n].get('weight', 1) for n in Graph.nodes()]
min_nw, max_nw = min(node_weights), max(node_weights)
node_sizes = [20 + 120 * (w - min_nw) / (max_nw - min_nw) for w in node_weights]
nx.draw_networkx_nodes(Graph, pos, node_size=node_sizes, node_color=node_colors)

# Labels for notable stations only
node_labels = {
    node: data['station']
    for node, data in Graph.nodes(data=True)
    if len(Graph.nodes) < 100 or 'Gangnam' in data['station']
}
nx.draw_networkx_labels(Graph, pos, labels=node_labels, font_size=8)

# --- 4. Legend ---
handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=cmap(i), markersize=10, label=f'Cluster {i}')
    for i in range(n_clusters)
]
plt.legend(handles=handles, title="Clusters", loc='upper left', framealpha=0.8)

plt.title("Seoul Subway Network — Spectral Clustering")
plt.axis('off')
plt.tight_layout()
plt.show()