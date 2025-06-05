from typing import Any

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger

import netquant
from netquant.utils.env_utils import log_gpu_memory_metadata
from netquant.utils.metadata_utils import log_metadata
from netquant.utils.misc import (
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_plugins,
    log_hyperparameters,
    register_custom_resolvers,
    task_wrapper,
)
from netquant.utils.pylogger import get_pylogger
from netquant.utils.saving_utils import save_state_dicts

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": netquant.CONFIG_ROOT.resolve().as_posix(),
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

    log_gpu_memory_metadata()

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        log.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.dataset._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset, _recursive_=False)

    # Init lightning model
    log.info(f"Instantiating lightning model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, fold_index=datamodule.fold_index, _recursive_=False)

    # Init callbacks
    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Init loggers
    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    # Init lightning ddp plugins
    log.info("Instantiating plugins...")
    plugins: list[Any] | None = instantiate_plugins(cfg)

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        plugins=plugins,
    )

    # Send parameters from cfg to all lightning loggers
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Log metadata
    log.info("Logging metadata!")
    log_metadata(cfg)

    # Train the model
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

    train_metrics = trainer.callback_metrics

    # Test the model
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # Save state dicts for best and last checkpoints
    if cfg.get("save_state_dict"):
        log.info("Starting saving state dicts!")
        save_state_dicts(
            trainer=trainer,
            model=model,
            dirname=cfg.paths.output_dir,
            **cfg.extras.state_dict_saving_params,
        )

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    # train the model
    _, _ = train(cfg)


if __name__ == "__main__":
    main()
