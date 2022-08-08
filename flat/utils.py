import pickle
import torch
import torch.nn as nn
import logging
import numpy as np
from pathlib import Path


logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def save_pickle(data, file_path):
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    with open(str(input_file), "rb") as f:
        data = pickle.load(f)
    return data


def load_model(model, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    logging.info(f"Loading model from {str(model_path)} .")
    states = torch.load(model_path)
    state = states["state_dict"]
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        norm = np.finfo(vector.dtype).eps
    return vector / norm


class AverageMeter():
    """
    computes and stores the average and current value
    Examples:
        >>> loss = AverageMeter()
        >>> for step, batch in enumerate(train_data):
        >>>     pred = self.model(batch)
        >>>     raw_loss = self.metric(pred, target)
        >>>     loss.update(raw_loss.item(), n=1)
        >>> cur_loss = loss.avg
    """
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
