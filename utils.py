import random
import os
import math
import time
import numpy as np
import pickle
import torch
import logging


def get_logger(filename: str):
    """Creates and returns a logger that logs to both the console and a file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    # File handler
    file_handler = logging.FileHandler(f"{filename}.log")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    return logger


def seed_everything(seed: int):
    """Sets random seed for reproducibility across various libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Tracks and stores the average and current values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def as_minutes(s: int) -> str:
    """Converts seconds to a string in minutes and seconds."""
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s}s"


def time_since(since: float, percent: float) -> str:
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return f"{as_minutes(s)} (remain {as_minutes(rs)})"


def convert_all_1d(array: list) -> list:
    """Converts 0-dimensional arrays in a list to 1-dimensional arrays."""
    return [np.array([item]) if item.ndim == 0 else item for item in array]


def save_pickle(path: str, contents):
    """Saves contents to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(contents, f)


def load_pickle(path: str):
    """Loads contents from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
