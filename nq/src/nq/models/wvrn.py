import numpy as np
import torch
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator, ClassifierMixin
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix

from nq.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class WVRN(BaseEstimator, ClassifierMixin):
    def __init__(self, max_iter=100, tol=1e-3):
        self.max_iter = max_iter
        self.tol = tol
        self.graph_ = None
        self.classes_ = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, graph: Data, train_mask: np.ndarray):  # train_mask: boolean array of shape [n_nodes]
        # extract adjacency matrix
        graph = graph.to(self.device)
        self.graph_ = graph

        n = graph.num_nodes
        A = to_scipy_sparse_matrix(graph.edge_index, num_nodes=n)
        edge_weight = torch.ones(graph.edge_index.size(1), dtype=torch.float32, device=self.device)
        A = torch.sparse_coo_tensor(indices=graph.edge_index, values=edge_weight, size=(n, n))

        # labels
        y = graph.y.int()
        self.classes_ = torch.unique(y[train_mask])
        k = len(self.classes_)

        # init probability matrix
        self.proba_ = torch.zeros((n, k), dtype=torch.float, device=self.device)

        # set training node distributions
        for idx, c in enumerate(self.classes_):
            self.proba_[y == c, idx] = 1.0

        # initialize unlabeled uniformly
        unlabeled = ~train_mask
        self.proba_[unlabeled] = 1.0 / k

        # iterative update
        for it in range(self.max_iter):
            old = self.proba_.clone()
            self.proba_ = torch.sparse.mm(A, self.proba_)
            # restore training labels
            self.proba_[train_mask] = old[train_mask]
            if torch.norm(self.proba_ - old) < self.tol:
                log.info(f"Converged after {it + 1} iterations.")
                break

        self.proba_ = self.proba_.cpu().numpy()
        self.graph_ = self.graph_.cpu()
        self.classes_ = self.classes_.cpu().numpy()
        return self

    def predict_proba(self, X):
        return self.proba_[X, :]

    def predict(self, X):
        return np.argmax(self.proba_[X, :], axis=1)

    def to_quapy_data(self, fold_index):
        quapy_data = {}
        graph = self.graph_.cpu()

        for stage in ["train", "val", "cal", "test"]:
            mask = graph[stage + "_mask"].cpu().numpy()

            instances = np.arange(graph.num_nodes)[mask[fold_index]]
            labels = graph.y.int()[mask[fold_index]].numpy()
            quapy_data[stage] = LabelledCollection(instances, labels)

        return quapy_data
