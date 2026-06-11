"""Mixup baseline + submit-time post-processing (priors, t-smoothing,
sonotype mirroring, rare-class suppression). Training stays unchanged."""
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
    # Per-stage flags (PRIORS_ENABLED, TEMPSMOOTH_ENABLED, ...) keep
    # their BaseCFG defaults (all on).

    TRAIN_CKPT_PATH = Path("./best_model_postproc.pt")
    LAST_CKPT_PATH  = Path("./last_model_postproc.pt")
    SUBMIT_CKPT_PATHS = [
        Path("/kaggle/input/birdclef-postproc/best_model_postproc.pt"),
        Path("/kaggle/input/birdclef-postproc/last_model_postproc.pt"),
    ]


if __name__ == "__main__":
    if CFG.MODE == "train":
        train(CFG)
    elif CFG.MODE == "submit":
        submit(CFG)
    else:
        raise SystemExit(f"Unknown CFG.MODE: {CFG.MODE!r}")
