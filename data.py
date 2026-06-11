"""Audio I/O, label maps, dataframes, datasets."""
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset


def parse_time_to_seconds(t):
    if isinstance(t, (int, float)):
        return float(t)
    s = str(t).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    return float(s)


def get_label_map(taxonomy_csv):
    tax = pd.read_csv(taxonomy_csv)
    labels = tax["primary_label"].astype(str).tolist()
    return labels, {lab: i for i, lab in enumerate(labels)}


def get_class_name_map(taxonomy_csv):
    tax = pd.read_csv(taxonomy_csv).set_index("primary_label")
    return tax["class_name"].astype(str).to_dict()


_FNAME_RE = re.compile(
    r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})(?:\.ogg|\.wav)?"
)


def parse_fname(name):
    """Extract site (S##) and UTC hour (0..23) from a BC2026 filename."""
    m = _FNAME_RE.match(name)
    if not m:
        return {"site": "unknown", "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}


def _read_window_from_open(f, start_frame, n_frames, path):
    if f.channels != 1:
        raise ValueError(f"Expected mono WAV, got {f.channels} channels: {path}")
    total = f.frames
    if start_frame >= total:
        return np.zeros(n_frames, dtype=np.float32)
    f.seek(start_frame)
    wave = f.read(frames=min(n_frames, total - start_frame), dtype="float32")
    if len(wave) < n_frames:
        wave = np.pad(wave, (0, n_frames - len(wave)), mode="constant")
    return wave


def read_wav_window(path, start_frame, n_frames):
    with sf.SoundFile(str(path)) as f:
        return _read_window_from_open(f, start_frame, n_frames, path)


def pick_crop_start(total, n_samples):
    if total <= n_samples:
        return 0
    return int(torch.randint(0, total - n_samples + 1, (1,)).item())


def build_focal_df(cfg, label_to_idx):
    df = pd.read_csv(cfg.TRAIN_CSV)

    df["file_path"] = df["filename"].apply(
        lambda n: str(cfg.TRAIN_AUDIO_DIR / n.replace(".ogg", ".wav"))
    )
    df["start_sec"] = np.nan
    df["group"]     = df["filename"]

    N = len(df)
    labels_arr = np.zeros((N, cfg.NUM_CLASSES), dtype=np.float32)
    primaries = df["primary_label"].astype(str).values
    secondaries_raw = df["secondary_labels"].values
    soft = float(cfg.SOFT_LABEL_COEFF)
    n_soft_set = 0
    for i in range(N):
        if primaries[i] in label_to_idx:
            labels_arr[i, label_to_idx[primaries[i]]] = 1.0
        if soft > 0.0:
            raw = secondaries_raw[i]
            if pd.isna(raw):
                continue
            try:
                sec_list = ast.literal_eval(str(raw))
            except (ValueError, SyntaxError):
                continue
            for lab in sec_list:
                lab = str(lab).strip()
                if lab and lab in label_to_idx:
                    j = label_to_idx[lab]
                    if soft > labels_arr[i, j]:
                        labels_arr[i, j] = soft
                        n_soft_set += 1
    if soft > 0.0:
        print(f"  soft secondary labels written: {n_soft_set:,} "
              f"(coeff={soft:.2f})")

    keep = ["file_path", "start_sec", "group"]
    return df[keep].reset_index(drop=True), labels_arr


def build_soundscape_df(cfg, label_to_idx):
    df = pd.read_csv(cfg.LABELS_CSV).drop_duplicates().reset_index(drop=True)

    df["file_path"] = df["filename"].apply(
        lambda n: str(cfg.SOUNDSCAPE_DIR / n.replace(".ogg", ".wav"))
    )
    df["start_sec"] = df["start"].apply(parse_time_to_seconds)
    df["group"]     = df["filename"]

    N = len(df)
    labels_arr = np.zeros((N, cfg.NUM_CLASSES), dtype=np.float32)
    for i, label_str in enumerate(df["primary_label"].astype(str).values):
        for lab in label_str.split(";"):
            lab = lab.strip()
            if lab in label_to_idx:
                labels_arr[i, label_to_idx[lab]] = 1.0

    keep = ["file_path", "start_sec", "group"]
    return df[keep].reset_index(drop=True), labels_arr


class BirdDataset(Dataset):
    """Focal recordings (random crop) and soundscape segments (fixed crop)
    through one __getitem__. A NaN start_sec means 'random crop from
    anywhere in the file'."""

    def __init__(self, cfg, df, labels_arr, train):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.labels = labels_arr
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["file_path"]

        if pd.isna(row["start_sec"]):
            with sf.SoundFile(path) as f:
                total = f.frames
                if self.train:
                    start = pick_crop_start(total, self.cfg.N_SAMPLES)
                else:
                    start = max(0, (total - self.cfg.N_SAMPLES) // 2)
                wave = _read_window_from_open(f, start, self.cfg.N_SAMPLES, path)
        else:
            start_frame = int(row["start_sec"] * self.cfg.SAMPLE_RATE)
            wave = read_wav_window(path, start_frame, self.cfg.N_SAMPLES)

        target = self.labels[idx]
        return torch.from_numpy(wave), torch.from_numpy(target)


class SoundscapeCropDataset(Dataset):
    """Extra training samples cut from labelled train soundscapes.

    Two modes (selected by CFG.SS_CROP_MODE):
      "random" — virtual epoch of N samples; each __getitem__ returns
                 a fresh random crop drawn from a random file.
      "dense"  — deterministic enumeration of every crop at a fixed
                 stride over each file's labelled span.

    Per-species label is the overlap-weighted sum of the segments the
    crop touches, clipped to [0, 1]:
      label[s] = clip( Σ_seg overlap_sec(crop, seg) / 5 , 0, 1 )   s ∈ seg
    """

    def __init__(self, cfg, label_to_idx):
        df = pd.read_csv(cfg.LABELS_CSV).drop_duplicates().reset_index(drop=True)
        df["start_sec"] = df["start"].apply(parse_time_to_seconds)
        df["end_sec"]   = df["end"].apply(parse_time_to_seconds)

        self.cfg       = cfg
        self.mode      = cfg.SS_CROP_MODE
        self.crop_dur  = cfg.DURATION
        self.n_classes = cfg.NUM_CLASSES

        self.files = []
        self.index = []   # used when mode == "dense"
        for fn, g in df.groupby("filename"):
            g = g.sort_values("start_sec")
            segs = []
            for _, row in g.iterrows():
                lv = np.zeros(cfg.NUM_CLASSES, dtype=np.float32)
                for lab in str(row["primary_label"]).split(";"):
                    lab = lab.strip()
                    if lab in label_to_idx:
                        lv[label_to_idx[lab]] = 1.0
                segs.append((float(row["start_sec"]),
                             float(row["end_sec"]), lv))
            if not segs:
                continue
            span_start = segs[0][0]
            span_end   = segs[-1][1]
            fidx = len(self.files)
            self.files.append({
                "path":       str(cfg.SOUNDSCAPE_DIR / fn.replace(".ogg", ".wav")),
                "segs":       segs,
                "span_start": span_start,
                "span_end":   span_end,
            })
            if self.mode == "dense":
                max_start = span_end - self.crop_dur
                if max_start <= span_start:
                    self.index.append((fidx, span_start))
                else:
                    t = span_start
                    while t <= max_start + 1e-6:
                        self.index.append((fidx, float(t)))
                        t += cfg.SS_CROP_STRIDE_SEC

        if not self.files:
            raise SystemExit(
                f"SoundscapeCropDataset: no usable files parsed from "
                f"{cfg.LABELS_CSV}."
            )

    def __len__(self):
        if self.mode == "dense":
            return len(self.index)
        return int(self.cfg.SS_CROP_N_SAMPLES)

    def _label_for_crop(self, segs, t):
        crop_s, crop_e = t, t + self.crop_dur
        label = np.zeros(self.n_classes, dtype=np.float32)
        for (s, e, lv) in segs:
            overlap = min(crop_e, e) - max(crop_s, s)
            if overlap > 0.0:
                label += (overlap / self.crop_dur) * lv
        np.clip(label, 0.0, 1.0, out=label)
        return label

    def __getitem__(self, idx):
        if self.mode == "dense":
            fidx, t = self.index[idx]
            f = self.files[fidx]
        else:
            f = self.files[np.random.randint(len(self.files))]
            span_s, span_e = f["span_start"], f["span_end"]
            max_start = span_e - self.crop_dur
            if max_start <= span_s:
                t = span_s
            else:
                t = float(np.random.uniform(span_s, max_start))

        start_frame = int(round(t * self.cfg.SAMPLE_RATE))
        wave = read_wav_window(Path(f["path"]), start_frame, self.cfg.N_SAMPLES)
        label = self._label_for_crop(f["segs"], t)
        return torch.from_numpy(wave), torch.from_numpy(label)
