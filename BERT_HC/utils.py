from os import listdir, path
from glob import glob
from datetime import datetime
from subprocess import Popen, PIPE, run
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict
import pandas as pd
import torch
import logging

HOME_DIR = str(Path.home())
INIT_TIME = datetime.now().strftime('%e-%m-%y_%H-%M-%S').lstrip()


def init_logger(name=None, path=None, screen=True):
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('{asctime} - {message}', datefmt="%H:%M:%S", style="{")
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(f"{path}/{name}-{INIT_TIME}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if screen:
        screen_handler = logging.StreamHandler()
        screen_handler.setFormatter(formatter)
        logger.addHandler(screen_handler)
    return logger


def count_num_cpu_gpu():
    if torch.cuda.is_available():
        num_gpu_cores = torch.cuda.device_count()
        num_cpu_cores = (cpu_count() // num_gpu_cores // 2) - 1
    else:
        num_gpu_cores = 0
        num_cpu_cores = (cpu_count() // 2) - 1
    return num_cpu_cores, num_gpu_cores


def save_predictions(folder, sample_idx_list, predictions_list, true_list, correct_list, class_probs, name):
    df_dict = {
        "sample_index": sample_idx_list,
        "prediction": predictions_list,
        "true": true_list,
        "correct": correct_list,
    }
    df_dict.update({f"class_{i}_prob": class_i_prob for i, class_i_prob in enumerate(class_probs)})
    df = pd.DataFrame.from_dict(df_dict)
    df = df.set_index("sample_index").sort_index()
    df.to_csv(f"{folder}/{name}-predictions.csv")


def get_checkpoint_file(ckpt_dir: str):
    for file in sorted(listdir(ckpt_dir)):
        if file.endswith(".ckpt"):
            return f"{ckpt_dir}/{file}"
    else:
        return None
