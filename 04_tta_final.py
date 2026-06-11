"""Adds multi-crop TTA inside the submit prediction loop. This is the
configuration used for the final submission."""
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

    POSTPROC_ENABLED = True

    TTA_ENABLED = True
    TTA_HOP_SEC = 2.5
    TTA_AGG     = "mean"

    TRAIN_CKPT_PATH = Path("./best_model_tta.pt")
    LAST_CKPT_PATH  = Path("./last_model_tta.pt")
    SUBMIT_CKPT_PATHS = [
        Path("/kaggle/input/birdclef-tta/best_model_tta.pt"),
        Path("/kaggle/input/birdclef-tta/last_model_tta.pt"),
    ]


if __name__ == "__main__":
    if CFG.MODE == "train":
        train(CFG)
    elif CFG.MODE == "submit":
        submit(CFG)
    else:
        raise SystemExit(f"Unknown CFG.MODE: {CFG.MODE!r}")
