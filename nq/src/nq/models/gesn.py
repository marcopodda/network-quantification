import torch
from graphesn import StaticGraphReservoir, initializer
from graphesn.util import approximate_graph_spectral_radius, to_sparse_adjacency
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


def get_sparse(graph):
    return to_sparse_adjacency(graph.edge_index, num_nodes=graph.num_nodes).t()


def get_alpha(sparse_adj_matrix):
    return approximate_graph_spectral_radius(sparse_adj_matrix)


def initialize_reservoir(graph, params, alpha, device="cpu"):
    reservoir = StaticGraphReservoir(
        num_layers=1,
        in_features=graph.num_node_features,
        hidden_features=params["gesn__hidden_features"],
        max_iterations=params["gesn__max_iterations"],
        bias=True,
    )
    reservoir.initialize_parameters(
        recurrent=initializer("uniform", rho=(params["gesn__recurrent_scale"] / alpha)),
        input=initializer("uniform", scale=params["gesn__input_scale"]),
        bias=initializer("uniform", scale=0.1),
    )
    reservoir.to(device)

    return reservoir


class GESN(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_features=1024,
        max_iterations=30,
        recurrent_scale=12.0,
        input_scale=0.5,
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.hidden_features = hidden_features
        self.max_iterations = max_iterations
        self.recurrent_scale = recurrent_scale
        self.input_scale = input_scale
        self.device = device
        self.max_iter = max_iter
        self.solver = solver
        self.C = C
        self.classifier_ = None
        self.is_fit = False
        self.embeddings_ = None
        self.graph_ = None
        self.classes_ = None

    def get_embeddings(self, graph):
        reservoir_params = {
            "gesn__hidden_features": self.hidden_features,
            "gesn__max_iterations": self.max_iter,
            "gesn__recurrent_scale": self.recurrent_scale,
            "gesn__input_scale": self.input_scale,
        }

        adj = get_sparse(graph.to(self.device))
        alpha = get_alpha(adj)

        reservoir = initialize_reservoir(graph, reservoir_params, alpha, self.device)
        embeddings = reservoir(adj.to(self.device), graph.x.to(self.device))

        if self.device == "cuda":
            torch.cuda.empty_cache()
            embeddings = embeddings.detach().cpu()

        return embeddings.numpy()

    def fit(self, graph, train_mask):
        self.graph_ = graph
        self.embeddings_ = self.get_embeddings(graph)
        self.classifier_ = LogisticRegression(max_iter=self.max_iter, solver=self.solver, C=self.C, n_jobs=-5)
        self.classifier_.fit(self.embeddings_[train_mask], graph.y.int()[train_mask].cpu().numpy())
        self.is_fit = True
        self.classes_ = self.classifier_.classes_
        self.num_classes_ = len(self.classes_)

    def predict(self, X):
        assert self.is_fit is True
        return self.classifier_.predict(X)

    def predict_proba(self, X):
        assert self.is_fit is True
        return self.classifier_.predict_proba(X)

    def to_quapy_data(self, fold_index):
        quapy_data = {}

        graph = self.graph_.cpu()

        for stage in ["train", "val", "cal", "test"]:
            mask = graph[stage + "_mask"]

            instances = self.embeddings_[mask[fold_index]]
            labels = graph.y.int()[mask[fold_index]].cpu().numpy()
            quapy_data[stage] = LabelledCollection(instances, labels)

        return quapy_data
