import quapy as qp


def to_quapy_collection(graph_embeddings, labels, mask=None):
    if mask is None:
        instances = graph_embeddings.to("cpu").numpy()
        labels = labels.to("cpu").numpy()
    else:
        instances = graph_embeddings[mask].to("cpu").numpy()
        labels = labels[mask].to("cpu").numpy()
    return qp.data.LabelledCollection(instances, labels)
