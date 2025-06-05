from pathlib import Path

import hydra
import joblib
import numpy as np
import quapy as qp
import torch
import yaml
from omegaconf import DictConfig
from quapy.protocol import APP, UPP

from nq import settings
from nq.quantification import METHODS
from nq.utils.misc import seed_everything, task_wrapper
from nq.utils.pylogger import get_pylogger

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": settings.CONFIG_DIR.resolve().as_posix(),
    "config_name": "quantify.yaml",
}
log = get_pylogger(__name__)


def quantify_parallel(data, protocol, method, model, task):
    qp.environ["SAMPLE_SIZE"] = 500
    qp.environ["N_JOBS"] = 5

    eval_data = data["test"] if task == "test" else data["val"]

    quant = METHODS[method](model)

    fit_params = {"fit_classifier": False}
    if method in ["ACC", "PACC", "HDy", "DMTopsoe", "EMQ"]:
        fit_params["val_split"] = data["cal"]
    quant.fit(data["train"], **fit_params)

    # evaluation
    protocol = protocol(eval_data, repeats=100, random_state=0)
    report = qp.evaluation.evaluation_report(quant, protocol=protocol, error_metrics=["mae", "mrae"])

    try:
        rae = np.mean(report["mrae"]).item()
        mae = np.mean(report["mae"]).item()
    except AttributeError:
        rae = np.nan
        mae = np.nan

    return {
        method: {
            "rae": rae,
            "mae": mae,
            "train_prev": data["train"].prevalence(),
            "true_prev": report["true-prev"],
            "estim_prev": report["estim-prev"],
            "report": report,
        }
    }


def get_scores_for_metric(results: dict, metric: str):
    scores = {}
    for key in results:
        scores[key] = results[key][metric]
    return scores


@task_wrapper
def quantify(cfg: DictConfig) -> tuple[dict | None, dict | None]:
    if Path("results.pt").exists():
        log.warning("Results file already exists. Skipping quantification.")
        return

    assert cfg.task_name in ["test", "validation"], f"{cfg.task_name=} is not supported!"

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        log.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    log.info(f"Running in <{cfg.task_name}> mode (trial={cfg.trial})")

    log.info(f"Instantiating dataset <{cfg.dataset._target_}>")
    dataset = hydra.utils.instantiate(cfg.dataset)

    # Init lightning model
    log.info(f"Loading model <{cfg.model}>")
    model_path = Path(".").resolve().parent.parent / "model.pt"
    model = torch.load(model_path, weights_only=False)
    log.info("Model loaded.")

    data = model.to_quapy_data(cfg.fold_index)
    log.info(f"Data loaded for fold {cfg.fold_index}")

    Protocol = APP if dataset.num_classes <= 2 else UPP
    log.info(f"Using protocol {Protocol} ({dataset.num_classes=})")

    methods = list(METHODS.keys())

    if Protocol == UPP and "HDy" in methods:
        methods.remove("HDy")

    if cfg.model in ["cdq", "enq"]:
        methods = ["ACC", "CC"]

    P = joblib.Parallel(n_jobs=len(methods), verbose=1)
    func = joblib.delayed(quantify_parallel)
    results = P(func(data, Protocol, m, model, cfg.task_name) for m in methods)

    log.info("Saving results.")

    # collate results
    results = {k: v for d in results for k, v in d.items()}
    torch.save(results, "results.pt")

    maes = get_scores_for_metric(results, "mae")
    with open("mae.yaml", "w") as file:
        yaml.dump(maes, file, Dumper=yaml.Dumper)

    raes = get_scores_for_metric(results, "rae")
    with open("rae.yaml", "w") as file:
        yaml.dump(raes, file, Dumper=yaml.Dumper)


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    # train the model
    quantify(cfg)


if __name__ == "__main__":
    main()
