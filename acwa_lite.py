## Accumulated Walks Embedding

import numpy as np
import networkx as nx
from CTDNE import CTDNE

"""ref: 
https://github.com/LogicJake/CTDNE/tree/master
[Nguyen, Giang Hoang, et al. "Continuous-time dynamic network embeddings." 3rd International Workshop on Learning Representations for Big Networks (WWW BigNet). 2018.]
"""

def set_seed(seed):
    np.random.seed(seed)

def ctdne_emb(graph, d, walk_length=20, num_walks=100):
    """Perform random walk and learn embeddings using CTDNE."""
    CTDNE_model = CTDNE(graph, dimensions=d, walk_length=walk_length, num_walks=num_walks, workers=4)
    model = CTDNE_model.fit(window=10, min_count=1, batch_words=4)
    return model

def process_snapshots(edges_list, num_nodes, dimensions, dataset_name, seed):
    """Process snapshots of a dynamic graph and compute embeddings."""
    set_seed(seed)
    accumulated_edges = np.array([[num_nodes - 1, num_nodes - 1]])  # Placeholder for max index node
    embeddings = []

    for i, edges in enumerate(edges_list):
        accumulated_edges = np.vstack([accumulated_edges, edges])
        current_nodes = np.unique(accumulated_edges.flatten())
        graph = nx.from_edgelist(accumulated_edges)

        # Assign random times to edges for CTDNE
        num_edges = len(graph.edges())
        edge_times = {edge: time for time, edge in enumerate(graph.edges())}
        nx.set_edge_attributes(graph, edge_times, 'time')
        
        # Generate embeddings
        model = ctdne_emb(graph, d=dimensions)
        emb_matrix = np.zeros((num_nodes, dimensions))
        
        for node in current_nodes:
            if str(node) in model.wv.key_to_index:
                emb_matrix[node] = model.wv[str(node)]
        
        embeddings.append(emb_matrix)
        print(f"Snapshot {i} processed.")

    # Save embeddings
    output_path = f"acwa{dimensions}/{dataset_name}_sd{seed}.npy"
    np.save(output_path, np.array(embeddings))
    print(f"Embeddings saved to {output_path}")


# Example Usage
if __name__ == "__main__":
    # Example parameters
    edges_list = [...]  # Replace with your edges list for snapshots
    num_nodes = 100000  # Replace with your number of nodes
    dimensions = 16
    dataset_name = "example_dataset"
    seed = 42

    process_snapshots(edges_list, num_nodes, dimensions, dataset_name, seed)




## Walk Alignment
from scipy.linalg import orthogonal_procrustes
import copy


def seen_nodes_accum(edges):
    """Compute cumulative seen nodes and currently inactive nodes at each snapshot."""
    all_v_accum = []
    all_seen_inactive_nodes = []
    current_v_accum = set()

    for i, a in enumerate(edges):
        current_v = set(np.unique(a.flatten()))
        current_seen_inactive = current_v_accum - current_v  # Nodes seen before but inactive now
        current_v_accum.update(current_v)
        all_v_accum.append(copy.copy(current_v_accum))
        all_seen_inactive_nodes.append(current_seen_inactive)
        print(f"Snapshot: {i:2d} | Seen Nodes: {len(current_v_accum):4d}  ||  Current Seen Inactive Nodes: {len(current_seen_inactive):4d}")

    return all_v_accum, all_seen_inactive_nodes


def alignment(node2vec_feats_raw, edges, lentrain):
    """Align embeddings using orthogonal Procrustes transformation."""
    T, N, d = node2vec_feats_raw.shape
    all_v_accum, all_seen_inactive_nodes = seen_nodes_accum(edges)
    print("OP with last snapshot of train set")
    embs_op = rwop_last_trained(node2vec_feats_raw, all_seen_inactive_nodes, lentrain)
    return embs_op


def rwop_last_trained(raw, all_seen_inactive_nodes, lentrain):
    """Perform recursive alignment of embeddings using the last trained snapshot as reference."""
    embs_op = copy.deepcopy(raw)

    # Align training snapshots from last train snapshot to first train snapshot
    for i in range(lentrain - 1, -1, -1):
        nodes_ref = all_seen_inactive_nodes[i + 1]
        nodes_current = all_seen_inactive_nodes[i]
        common_nodes = list(nodes_ref & nodes_current)
        
        if common_nodes:
            print(f"Snapshot {i:2d} -> {i+1:2d} | Common Nodes: {len(common_nodes):4d}")
            e_ref = embs_op[i + 1][common_nodes]
            e_trans_before = raw[i][common_nodes]

            R, _ = orthogonal_procrustes(e_trans_before, e_ref)
            embs_op[i] = np.dot(raw[i], R)

    # Align test snapshots from last train snapshot to last test snapshot
    for i in range(lentrain + 1, len(raw)):
        nodes_ref = all_seen_inactive_nodes[i - 1]
        nodes_current = all_seen_inactive_nodes[i]
        common_nodes = list(nodes_ref & nodes_current)

        if common_nodes:
            print(f"Snapshot {i-1:2d} -> {i:2d} | Common Nodes: {len(common_nodes):4d}")
            e_ref = embs_op[i - 1][common_nodes]
            e_trans_before = raw[i][common_nodes]

            R, _ = orthogonal_procrustes(e_trans_before, e_ref)
            embs_op[i] = np.dot(raw[i], R)

    return embs_op
