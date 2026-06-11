"""Adds classic 2-sample mixup on top of the baseline. KD teacher runs
on the mixed waveform."""
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

    TRAIN_CKPT_PATH = Path("./best_model_mixup.pt")
    LAST_CKPT_PATH  = Path("./last_model_mixup.pt")
    SUBMIT_CKPT_PATHS = [
        Path("/kaggle/input/birdclef-mixup/best_model_mixup.pt"),
        Path("/kaggle/input/birdclef-mixup/last_model_mixup.pt"),
    ]


if __name__ == "__main__":
    if CFG.MODE == "train":
        train(CFG)
    elif CFG.MODE == "submit":
        submit(CFG)
    else:
        raise SystemExit(f"Unknown CFG.MODE: {CFG.MODE!r}")
