import os
import random
import time
from typing import Callable

import numpy as np
import torch
from nq.utils.pylogger import get_pylogger
from omegaconf import DictConfig

log = get_pylogger(__name__)


def seed_everything(seed: int = 42):
    """
    Seed all major libraries and settings to ensure reproducibility.

    Parameters
    ----------
    seed : int
        The random seed to set.
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Torch backend settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set PYTHONHASHSEED environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
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
            timer = time.time()
            task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            raise ex

        # things to always do after either success or exception
        finally:
            # print time elapsed
            output_file = "time.txt"
            with open(output_file, "w") as ofile:
                print(f"{time.time() - timer}", file=ofile)

            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

    return wrap


def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup).

    Args:
        path (str): File path.
        content (str): File content.
    """

    with open(path, "w+") as file:
        file.write(content)
