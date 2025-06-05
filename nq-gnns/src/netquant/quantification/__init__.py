from netquant.datasets.cora import Cora
from netquant.datasets.genius import Genius
from netquant.datasets.questions import Questions
from netquant.datasets.toloker import Toloker
from netquant.datasets.twitch import Twitch
from pandas.api.types import CategoricalDtype
from quapy.method.aggregative import ACC, CC, EMQ, PACC, PCC, DistributionMatching, HDy


class Topsoe(DistributionMatching):
    def __init__(self, classifier):
        super().__init__(classifier, divergence="topsoe")


DATASETS = {
    "cora-0-vs-all": Cora,
    "cora-1-vs-all": Cora,
    "cora-2-vs-all": Cora,
    "cora-3-vs-all": Cora,
    "cora-4-vs-all": Cora,
    "cora-5-vs-all": Cora,
    "cora-6-vs-all": Cora,
    "genius": Genius,
    "questions": Questions,
    "toloker": Toloker,
    "twitch": Twitch,
}
DATASETS_CAT = CategoricalDtype(categories=DATASETS.keys(), ordered=True)


METHODS = {
    "CC": CC,
    "ACC": ACC,
    "PCC": PCC,
    "PACC": PACC,
    "HDy": HDy,
    "DMTopsoe": Topsoe,
    "EMQ": EMQ,
}
METHODS_CAT = CategoricalDtype(categories=METHODS.keys(), ordered=True)

MODELS = ["gcn", "gat", "gin"]
MODELS_CAT = CategoricalDtype(categories=MODELS, ordered=True)
