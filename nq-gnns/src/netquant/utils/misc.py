import argparse
import time
import warnings
from functools import wraps
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

from netquant.utils.pylogger import get_pylogger
from netquant.utils.rich_utils import print_config_tree

log = get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished or failed
    - Logging the exception if occurs
    - Logging the output dir

    Args:
        task_func (Callable): Task function.

    Returns:
        Callable: Decorator that wraps the task function in extra utilities.
    """

    def wrap(cfg: DictConfig):
        # execute the task
        timer = 0
        try:
            # apply extra utilities
            extras(cfg)

            timer = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # when using hydra plugins like Optuna, you might want to disable
            # raising exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # print time elapsed
            output_file = Path(cfg.paths.output_dir) / "time.txt"
            with open(output_file, "w") as ofile:
                print(f"{time.time() - timer}", file=ofile)

            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # close loggers (even if exception occurs so multirun won't fail)
            close_loggers()

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Rich config printing

    Args:
        cfg (DictConfig): Main config.
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config.

    Args:
        callbacks_cfg (DictConfig): Callbacks config.

    Returns:
        list[Callback]: List with all instantiated callbacks.
    """

    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiates loggers from config.

    Args:
        logger_cfg (DictConfig): Loggers config.

    Returns:
        list[Logger]: List with all instantiated loggers.
    """

    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Saves additionally:
    - Number of model parameters

    Args:
        object_dict (dict): Dict object with all parameters.
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["dataset"] = cfg["dataset"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during
    multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb  # type:ignore

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup).

    Args:
        path (str): File path.
        content (str): File content.
    """

    with open(path, "w+") as file:
        file.write(content)


def instantiate_plugins(cfg: DictConfig) -> list[Any] | None:
    """Instantiates lightning plugins from config.

    Args:
        cfg (DictConfig): Config.

    Returns:
        list[Any]: List with all instantiated plugins.
    """

    if not cfg.extras.get("plugins"):
        log.warning("No plugins configs found! Skipping...")
        return

    if cfg.trainer.get("accelerator") == "cpu":
        log.warning("Using CPU as accelerator! Skipping...")
        return

    plugins: list[Any] = []
    for _, pl_conf in cfg.extras.get("plugins").items():
        if isinstance(pl_conf, DictConfig) and "_target_" in pl_conf:
            log.info(f"Instantiating plugin <{pl_conf._target_}>")
            plugins.append(hydra.utils.instantiate(pl_conf))

    return plugins


def get_args_parser() -> argparse.ArgumentParser:
    """Get parser for additional Hydra's command line flags."""
    parser = argparse.ArgumentParser(description="Additional Hydra's command line flags parser.")

    parser.add_argument(
        "--config-path",
        "-cp",
        nargs="?",
        default=None,
        help="""Overrides the config_path specified in hydra.main().
                    The config_path is absolute or relative to the Python file declaring @hydra.main()""",
    )

    parser.add_argument(
        "--config-name",
        "-cn",
        nargs="?",
        default=None,
        help="Overrides the config_name specified in hydra.main()",
    )

    parser.add_argument(
        "--config-dir",
        "-cd",
        nargs="?",
        default=None,
        help="Adds an additional config dir to the config search path",
    )
    return parser


def register_custom_resolvers(version_base: str, config_path: str, config_name: str) -> Callable:
    """Optional decorator to register custom OmegaConf resolvers. It is
    excepted to call before `hydra.main` decorator call.

    Args:
        version_base (str): Hydra version base.
        config_path (str): Hydra config path.
        config_name (str): Hydra config name.

    Returns:
        Callable: Decorator that registers custom resolvers before running
            main function.
    """

    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return function(*args, **kwargs)

        return wrapper

    return decorator
