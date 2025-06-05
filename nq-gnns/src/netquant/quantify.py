from pathlib import Path

import hydra
import numpy as np
import quapy as qp
import torch
import yaml
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from quapy.protocol import APP

import netquant
from netquant.quantification import METHODS
from netquant.quantification.selection import load_fold_data, load_graph, load_model
from netquant.quantification.utils import needs_calibration
from netquant.quantification.wrapper import WrapperEstimator
from netquant.utils.misc import register_custom_resolvers, task_wrapper
from netquant.utils.pylogger import get_pylogger

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": netquant.CONFIG_ROOT.resolve().as_posix(),
    "config_name": "quantify.yaml",
}
log = get_pylogger(__name__)


def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    """Saves numpy floats in readable format when saving a .yaml file."""
    return dumper.represent_list(array.tolist())


yaml.add_representer(np.ndarray, ndarray_representer)


@task_wrapper
def quantify(cfg: DictConfig) -> tuple[dict | None, dict | None]:
    """Performs quantifiction."""

    assert cfg.task_name in ["test", "validation"], f"{cfg.task_name=} is not supported!"

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        log.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    qp.environ["SAMPLE_SIZE"] = 500
    log.info(f"Setting SAMPLE_SIZE={qp.environ['SAMPLE_SIZE']}")

    qp.environ["N_JOBS"] = -1
    log.info(f"Setting N_JOBS={qp.environ['N_JOBS']}")

    log.info(f"Load model with cfg <{cfg}>")
    model = load_model(cfg)

    log.info(f"Load graph {cfg.dataset_name}")
    graph = load_graph(cfg.dataset_name)

    log.info("Obtaining graph embeddings...")
    embeddings = model.forward(graph.x, graph.edge_index)

    log.info(f"Getting data for fold {cfg.fold_index}")
    train, cal, val, test = load_fold_data(embeddings, graph, cfg.fold_index)
    eval_data = test if cfg.task_name == "test" else val

    results = {method: {} for method in METHODS}
    scores = {}

    for method in METHODS:
        log.info(f"Evaluating method <{method}>")
        estimator = WrapperEstimator(graph, model)
        quant = METHODS[method](estimator)

        fit_params = {"fit_classifier": False}
        if needs_calibration(method):
            fit_params["val_split"] = cal

        quant.fit(train, **fit_params)
        protocol = APP(eval_data, repeats=100, random_state=0)
        tp, ep = qp.evaluation.prediction(quant, protocol)
        report = qp.evaluation.evaluation_report(quant, protocol=protocol, error_metrics=["mae"])
        mae = qp.evaluation.evaluate(quant, protocol, error_metric="mae")
        scores[method] = float(mae)

        results[method] = {
            "mae": mae,
            "train_prev": train.prevalence(),
            "true_prev": tp,
            "estim_prev": ep,
            "report": report,
        }

        log.info(f"{method} MAE is {results[method]['mae']:.3f}")

    log.info("Saving results.")

    output_dir = Path(cfg.paths.output_dir)
    with open(output_dir / "scores.yaml", "w") as file:
        yaml.dump(scores, file, Dumper=yaml.Dumper)

    torch.save(results, output_dir / "results.pth")

    return None, None  # for compatibility with task_wrapper


@register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    # train the model
    quantify(cfg)


if __name__ == "__main__":
    main()
