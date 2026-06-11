"""Starting point: KD-on baseline. No mixup, no extra augment, no
post-processing, no TTA. Other scripts in this folder layer features on
top of this configuration."""
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
    MODE = "train"   # "train" or "submit"

    TRAIN_CKPT_PATH = Path("./best_model_baseline.pt")
    LAST_CKPT_PATH  = Path("./last_model_baseline.pt")
    SUBMIT_CKPT_PATHS = [
        Path("/kaggle/input/birdclef-baseline/best_model_baseline.pt"),
        Path("/kaggle/input/birdclef-baseline/last_model_baseline.pt"),
    ]


if __name__ == "__main__":
    if CFG.MODE == "train":
        train(CFG)
    elif CFG.MODE == "submit":
        submit(CFG)
    else:
        raise SystemExit(f"Unknown CFG.MODE: {CFG.MODE!r}")
