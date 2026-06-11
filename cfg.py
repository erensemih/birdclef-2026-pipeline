"""Base config. Iteration scripts subclass BaseCFG and override fields."""
from pathlib import Path
import torch


class BaseCFG:
    MODE = "train"   # "train" or "submit"

    # Paths (local training)
    BASE_DIR        = Path(".")
    WAV_DIR         = Path("birdclef-2026-wav")
    TRAIN_AUDIO_DIR = WAV_DIR / "train_audio_wav"
    SOUNDSCAPE_DIR  = WAV_DIR / "train_soundscapes_wav"
    TRAIN_CSV       = BASE_DIR / "train.csv"
    LABELS_CSV      = BASE_DIR / "train_soundscapes_labels.csv"
    TAXONOMY_CSV    = BASE_DIR / "taxonomy.csv"
    PERCH_PATH      = Path("./perch_v2")

    # Source OGG dirs — used by prepare_data.ensure_wav_dirs() to convert
    # to WAV on first run if the WAV dirs above are empty.
    OGG_BASE_DIR        = Path("birdclef-2026")
    OGG_TRAIN_AUDIO_DIR = OGG_BASE_DIR / "train_audio"
    OGG_SOUNDSCAPE_DIR  = OGG_BASE_DIR / "train_soundscapes"

    TRAIN_CKPT_PATH = Path("./best_model.pt")
    LAST_CKPT_PATH  = Path("./last_model.pt")

    # Audio
    SAMPLE_RATE   = 32000
    DURATION      = 5.0
    N_SAMPLES     = int(SAMPLE_RATE * DURATION)
    FILE_DURATION = 60.0
    N_OUTPUT_WIN  = 12

    # Mel spectrogram (256 x 256 native — no resize)
    N_MELS     = 256
    N_FFT      = 2048
    WIN_LENGTH = 626
    HOP_LENGTH = 627
    F_MIN      = 20
    F_MAX      = 16000
    TOP_DB     = 80.0
    LMS_SHAPE  = (256, 256)

    # Model
    MODEL_NAME  = "efficientnet_b0"
    PRETRAINED  = True
    NUM_CLASSES = 234
    DROP_PROB   = 0.3

    # Training
    EPOCHS           = 1
    WARMUP_EPOCHS    = 5
    BATCH_SIZE       = 64
    NUM_WORKERS      = 16
    LR               = 5e-4
    WEIGHT_DECAY     = 1e-4
    DIV_FACTOR       = 25
    FINAL_DIV_FACTOR = 1e4
    POS_WEIGHT_CLIP  = 25.0

    # KD from Perch v2
    KD_ENABLED      = True
    KD_LAMBDA       = 1.0
    PERCH_EMBED_DIM = 1536

    # Soft secondary labels (focal only)
    SOFT_LABEL_COEFF = 0.3

    # Mixup (off by default; each iteration script flips what it needs)
    MIXUP_ENABLED          = False
    MIXUP_MODE             = "convex"     # "convex" or "temporal_concat"
    MIXUP_PROB             = 0.5
    MIXUP_ALPHA            = 0.5
    MIXUP_CONCAT_PIECES    = 2
    MIXUP_CONCAT_MIN_SEC   = 0.5
    MIXUP_CONCAT_XFADE_SEC = 0.03

    # Extra soundscape crops added to training (optional)
    SS_CROP_MODE       = "off"     # "off", "random", or "dense"
    SS_CROP_N_SAMPLES  = 10000     # used when SS_CROP_MODE == "random"
    SS_CROP_STRIDE_SEC = 1.0       # used when SS_CROP_MODE == "dense"

    # Submit (Kaggle)
    KAGGLE_DATA_DIR       = Path("/kaggle/input/competitions/birdclef-2026")
    SUBMIT_CKPT_PATHS     = []     # filled in per script
    SUBMIT_OUT_CSV        = Path("submission.csv")
    SUBMIT_INFER_BATCH    = 32
    SUBMIT_FALLBACK_FILES = 650

    # Submit TTA
    TTA_ENABLED = False
    TTA_HOP_SEC = 2.5
    TTA_AGG     = "mean"           # "mean" (overlap-weighted) or "max"

    # Submit post-processing (master toggle + per-stage flags)
    POSTPROC_ENABLED      = False

    PRIORS_ENABLED        = True
    PRIORS_LAMBDA         = 0.4
    PRIORS_HOUR_SHRINK    = 8.0
    PRIORS_SITE_SHRINK    = 8.0
    PRIORS_SH_SHRINK      = 1.0

    TEMPSMOOTH_ENABLED    = True
    TEMPSMOOTH_DF         = 1.20
    TEMPSMOOTH_RADIUS     = 3
    TEMPSMOOTH_ALPHA      = 0.30

    SONOTYPE_GROUPS = (
        ("47158son15", "47158son16"),
        ("47158son09", "47158son12"),
        ("47158son02", "47158son14"),
        ("47158son13", "47158son21", "47158son22", "47158son23"),
    )

    RARE_SUPPRESS_ENABLED = True
    RARE_CLASS_NAMES      = ("Amphibia", "Mammalia", "Reptilia")
    RARE_MARGIN           = 0.05
    RARE_SCALE            = 0.9

    # Misc
    SEED    = 42
    DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = torch.cuda.is_available()
