from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class LR(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        device="cpu",
    ):
        self.device = device
        self.max_iter = max_iter
        self.solver = solver
        self.C = C
        self.classifier_ = None
        self.is_fit = False
        self.graph_ = None
        self.classes_ = None
        self.embeddings_ = None

    def fit(self, graph, train_mask):
        self.graph_ = graph
        self.embeddings_ = self.graph.x
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
