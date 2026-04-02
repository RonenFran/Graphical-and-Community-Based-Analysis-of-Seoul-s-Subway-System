# pip install node2vec umap-learn hdbscan
from node2vec import Node2Vec
import umap
import hdbscan

# --- 1. Learn Node2Vec embeddings ---
node2vec = Node2Vec(
    Graph,
    dimensions=64,       # embedding size
    walk_length=30,      # steps per random walk
    num_walks=200,       # walks per node
    p=1.0,               # return parameter (1 = neutral)
    q=0.5,               # in-out parameter (<1 = DFS-biased, explores communities)
    weight_key='weight', # use traffic weights to bias walks
    workers=4,
    quiet=True,
)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

node_list = list(Graph.nodes())
embeddings = np.array([model.wv[n] for n in node_list])  # shape: (N, 64)

# --- 2. UMAP: reduce to 2D (use as both cluster input and layout) ---
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42,
)
embedding_2d = reducer.fit_transform(embeddings)  # shape: (N, 2)

# Use UMAP coords as graph layout
pos = {node: embedding_2d[i] for i, node in enumerate(node_list)}

# --- 3. HDBSCAN clustering on 2D embedding ---
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,
    min_samples=3,
    metric='euclidean',
)
labels = clusterer.fit_predict(embedding_2d)
# Note: label == -1 means noise/outlier

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"HDBSCAN found {n_clusters} clusters, "
      f"{(labels == -1).sum()} outlier nodes")

cluster_map = {node: labels[i] for i, node in enumerate(node_list)}

# --- 4. Assign colors (outliers get gray) ---
cmap = cm.get_cmap('tab20', max(n_clusters, 1))
node_colors = [
    'lightgray' if cluster_map[n] == -1 else cmap(cluster_map[n])
    for n in Graph.nodes()
]

# --- 5. Draw ---
plt.figure(figsize=(14, 14))

weights = list(nx.get_edge_attributes(Graph, 'weight').values())
if weights:
    min_weight, max_weight = min(weights), max(weights)
    normalized_weights = [
        1 + (x - min_weight) / (max_weight - min_weight) * 4 for x in weights
    ]
else:
    normalized_weights = [1.0] * Graph.number_of_edges()

nx.draw_networkx_edges(Graph, pos, width=normalized_weights, alpha=0.4, edge_color='gray')

node_weights = [Graph.nodes[n].get('weight', 1) for n in Graph.nodes()]
min_nw, max_nw = min(node_weights), max(node_weights)
node_sizes = [20 + 120 * (w - min_nw) / (max_nw - min_nw) for w in node_weights]
nx.draw_networkx_nodes(Graph, pos, node_size=node_sizes, node_color=node_colors)

node_labels = {
    node: data['station']
    for node, data in Graph.nodes(data=True)
    if len(Graph.nodes) < 100 or 'Gangnam' in data['station']
}
nx.draw_networkx_labels(Graph, pos, labels=node_labels, font_size=8)

# --- 6. Legend ---
handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=cmap(i), markersize=10, label=f'Cluster {i}')
    for i in range(n_clusters)
]
handles.append(plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor='lightgray', markersize=10, label='Outlier'))
plt.legend(handles=handles, title="Clusters", loc='upper left', framealpha=0.8)

plt.title("Seoul Subway Network — Node2Vec + UMAP + HDBSCAN")
plt.axis('off')
plt.tight_layout()
plt.show()