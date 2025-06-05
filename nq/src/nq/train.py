from pathlib import Path

import hydra
import torch
from nq import settings
from nq.models import initialize_model
from nq.utils.misc import seed_everything, task_wrapper
from nq.utils.pylogger import get_pylogger
from omegaconf import DictConfig

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": settings.CONFIG_DIR.resolve().as_posix(),
    "config_name": "train.yaml",
}
log = get_pylogger(__name__)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best
    weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator which applies
    extra utilities before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated
        objects.
    """
    if Path("model.pt").exists():
        log.warning("Model file already exists. Skipping training.")
        return

    log.info(f"Using device: {torch.cuda.get_device_name(0)}")
    log.info(f"Torch available: {torch.cuda.is_available()}")

    # # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        log.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating dataset <{cfg.dataset._target_}>")
    dataset = hydra.utils.instantiate(cfg.dataset)

    # Init lightning model
    log.info(f"Instantiating model <{cfg.model}>")
    model = initialize_model(cfg.model, cfg.trial)

    log.info(f"Fitting model <{cfg.model}>")
    model.fit(dataset.graph, dataset.graph.train_mask[cfg.fold_index])
    log.info("Model fitted.")

    torch.save(model, "model.pt", pickle_protocol=5)
    log.info("Model saved.")


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    # train the model
    _, _ = train(cfg)


if __name__ == "__main__":
    main()
