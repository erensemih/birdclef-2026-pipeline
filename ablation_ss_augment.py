"""Ablation: random 5-sec crops from labelled train soundscapes are added
to the training set with overlap-weighted multi-hot labels (Option A).
Did not move the leaderboard. Kept for the record."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

from cfg import BaseCFG
from infer import submit
from loop import train


class CFG(BaseCFG):
    MODE = "train"

    MIXUP_ENABLED = True
    MIXUP_MODE    = "convex"
    MIXUP_PROB    = 0.5
    MIXUP_ALPHA   = 0.5

    SS_CROP_MODE      = "random"
    SS_CROP_N_SAMPLES = 10000

    POSTPROC_ENABLED = True
    TTA_ENABLED      = True
    TTA_HOP_SEC      = 2.5
    TTA_AGG          = "mean"

    TRAIN_CKPT_PATH = Path("./best_model_ss_augment.pt")
    LAST_CKPT_PATH  = Path("./last_model_ss_augment.pt")
    SUBMIT_CKPT_PATHS = [
        Path("/kaggle/input/birdclef-ss-augment/best_model_ss_augment.pt"),
        Path("/kaggle/input/birdclef-ss-augment/last_model_ss_augment.pt"),
    ]


if __name__ == "__main__":
    if CFG.MODE == "train":
        train(CFG)
    elif CFG.MODE == "submit":
        submit(CFG)
    else:
        raise SystemExit(f"Unknown CFG.MODE: {CFG.MODE!r}")
