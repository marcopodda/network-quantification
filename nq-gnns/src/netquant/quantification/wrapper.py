import numpy as np
import torch
from sklearn.base import BaseEstimator


class WrapperEstimator(BaseEstimator):
    def __init__(self, graph, model):
        self.graph = graph
        self.model = model
        self.classes_ = torch.unique(graph.y, sorted=True).numpy()
        self.num_classes = len(self.classes_)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        self.model.eval()
        X = torch.from_numpy(X)
        p = self.model.predict_embeddings(X).numpy()
        return np.concatenate([1 - p, p], axis=1)

    def predict(self, X):
        self.model.eval()
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1).reshape(-1)
