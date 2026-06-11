"""Ablation: dense (deterministic) soundscape crops + temporal-concat
mixup (K contiguous chunks with equal-power crossfade). The idea was
that two clips concatenated end-to-end sound more like a real ambient
recording than two clips overlaid. Did not move the leaderboard."""
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

    MIXUP_ENABLED          = True
    MIXUP_MODE             = "temporal_concat"
    MIXUP_PROB             = 0.5
    MIXUP_CONCAT_PIECES    = 2
    MIXUP_CONCAT_MIN_SEC   = 0.5
    MIXUP_CONCAT_XFADE_SEC = 0.03

    SS_CROP_MODE       = "dense"
    SS_CROP_STRIDE_SEC = 1.0

    POSTPROC_ENABLED = True
    TTA_ENABLED      = True
    TTA_HOP_SEC      = 2.5
    TTA_AGG          = "mean"

    TRAIN_CKPT_PATH = Path("./best_model_temporal_concat.pt")
    LAST_CKPT_PATH  = Path("./last_model_temporal_concat.pt")
    SUBMIT_CKPT_PATHS = [
        Path("/kaggle/input/birdclef-temporal-concat/best_model_temporal_concat.pt"),
        Path("/kaggle/input/birdclef-temporal-concat/last_model_temporal_concat.pt"),
    ]


if __name__ == "__main__":
    if CFG.MODE == "train":
        train(CFG)
    elif CFG.MODE == "submit":
        submit(CFG)
    else:
        raise SystemExit(f"Unknown CFG.MODE: {CFG.MODE!r}")
