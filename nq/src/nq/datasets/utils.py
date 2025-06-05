import torch
import torch.nn.functional as F
from torch_geometric.utils import homophily


def graph_statistics(graph):
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.num_edges}")
    print(f"Average node degree: {graph.num_edges / graph.num_nodes:.2f}")
    print(f"Number of training nodes: {graph.train_mask.sum()}")
    print(f"Number of val nodes: {graph.val_mask.sum()}")
    print(f"Number of test nodes: {graph.test_mask.sum()}")
    print(f"Training node label rate: {int(graph.train_mask.sum()) / graph.num_nodes:.2f}")
    print(f"Has isolated nodes: {graph.has_isolated_nodes()}")
    print(f"Has self-loops: {graph.has_self_loops()}")
    print(f"Is undirected: {graph.is_undirected()}")
    print(f"Homophily (edge, node, edge_insensitive): {compute_homophily(graph)}")


def transform_one_hot_encoding(y):
    num_classes = int(y.max()) + 1
    y = F.one_hot(y.type(torch.long), num_classes)
    return y


def compute_homophily(data, methods=["edge", "node", "edge_insensitive"]):
    homophily_values = []
    for method in methods:
        homophily_values.append(homophily(data.edge_index, data.y.long(), method=method))
    return homophily_values
