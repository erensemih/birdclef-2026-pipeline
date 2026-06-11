"""Mel extractor, EfficientNet-B0 student, AUC, seeding."""
import os
import random

import numpy as np
import timm
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import roc_auc_score


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class LogMelExtractor(nn.Module):
    """Power mel spectrogram → dB → clamp to [-TOP_DB, 0] → linear [0, 1]."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.SAMPLE_RATE, n_fft=cfg.N_FFT,
            win_length=cfg.WIN_LENGTH, hop_length=cfg.HOP_LENGTH,
            f_min=cfg.F_MIN, f_max=cfg.F_MAX, n_mels=cfg.N_MELS,
            power=2.0, center=True, pad_mode="reflect",
            norm="slaney", mel_scale="slaney",
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")
        n_frames_expected = 1 + cfg.N_SAMPLES // cfg.HOP_LENGTH
        assert n_frames_expected == cfg.LMS_SHAPE[1], (
            f"HOP_LENGTH={cfg.HOP_LENGTH} gives {n_frames_expected} time "
            f"frames; expected {cfg.LMS_SHAPE[1]}."
        )

    @torch.no_grad()
    def forward(self, wave):
        mel = self.mel(wave)
        lms = self.db(mel)
        lms = lms.clamp(min=-self.cfg.TOP_DB, max=0.0)
        lms = (lms + self.cfg.TOP_DB) / self.cfg.TOP_DB
        return lms.unsqueeze(1)


class BirdCLEFModel(nn.Module):
    """timm backbone (default EfficientNet-B0) + classifier head + optional
    KD projection head."""

    def __init__(self, cfg, with_proj=False):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.MODEL_NAME, pretrained=cfg.PRETRAINED,
            in_chans=1, num_classes=0, global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(cfg.DROP_PROB),
            nn.Linear(feat_dim, cfg.NUM_CLASSES),
        )
        self.proj_head = (
            nn.Sequential(nn.Linear(feat_dim, cfg.PERCH_EMBED_DIM))
            if with_proj else None
        )

    def forward(self, x, return_proj=False):
        feats = self.backbone(x)
        logits = self.head(feats)
        if return_proj:
            if self.proj_head is None:
                raise RuntimeError("return_proj=True needs with_proj=True.")
            with torch.amp.autocast(
                device_type=feats.device.type, enabled=False
            ):
                proj = self.proj_head(feats.float())
            return logits, proj
        return logits


def competition_macro_auc(targets, probs):
    """Macro-AUC over the classes that have at least one positive in
    targets — mirrors the official BirdCLEF+ metric."""
    y_true = (targets > 0.5).astype(np.int8)
    sums = y_true.sum(axis=0)
    scored = sums > 0
    n = int(scored.sum())
    if n == 0:
        return 0.0, 0
    auc = roc_auc_score(y_true[:, scored], probs[:, scored], average="macro")
    return float(auc), n
