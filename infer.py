"""Submit pipeline: per-file prediction (optionally with TTA), rank-average
ensemble across checkpoints, optional post-processing chain."""
import gc
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

from data import get_label_map
from model import BirdCLEFModel, LogMelExtractor
from postproc import apply_postproc_chain


@torch.no_grad()
def _predict_one_file(cfg, path, model, mel_extractor, device, batch_size):
    """Plain 12-window prediction over a 60-sec clip."""
    wave, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != cfg.SAMPLE_RATE:
        raise ValueError(f"Expected {cfg.SAMPLE_RATE} Hz, got {sr}: {path}")
    if wave.ndim != 1:
        raise ValueError(f"Expected mono, got shape {wave.shape}: {path}")

    n = cfg.N_SAMPLES
    segs = np.zeros((12, n), dtype=np.float32)
    for k in range(12):
        s = k * n
        e = min(s + n, len(wave))
        if e > s:
            segs[k, : e - s] = wave[s:e]

    out_chunks = []
    for i in range(0, 12, batch_size):
        batch = torch.from_numpy(segs[i:i + batch_size]).to(
            device, non_blocking=True
        )
        mel = mel_extractor(batch)
        logits = model(mel)
        out_chunks.append(torch.sigmoid(logits).float().cpu().numpy())
    probs = np.concatenate(out_chunks, axis=0)
    return probs, path.stem


def _aggregate_tta_to_official(all_probs, hop_sec, win_sec, n_output, agg):
    """Reduce N TTA windows to n_output 5-sec official positions by
    overlap-area-weighted mean (or max)."""
    N = len(all_probs)
    num_classes = all_probs.shape[1]
    out = np.zeros((n_output, num_classes), dtype=np.float32)
    for k in range(1, n_output + 1):
        out_start = win_sec * (k - 1)
        out_end   = win_sec * k
        idxs, weights = [], []
        for i in range(N):
            w_start = i * hop_sec
            w_end   = w_start + win_sec
            overlap = max(0.0, min(w_end, out_end) - max(w_start, out_start))
            if overlap > 0:
                idxs.append(i)
                weights.append(overlap)
        if not idxs:
            continue
        idxs = np.asarray(idxs, dtype=np.int64)
        if agg == "max":
            out[k - 1] = np.max(all_probs[idxs], axis=0)
        else:
            w = np.asarray(weights, dtype=np.float32)
            w = w / w.sum()
            out[k - 1] = (w[:, None] * all_probs[idxs]).sum(axis=0)
    return out


@torch.no_grad()
def _predict_one_file_tta(cfg, path, model, mel_extractor, device, batch_size):
    """Slide 5-sec windows at CFG.TTA_HOP_SEC, predict each, then aggregate
    back to the 12 official output positions."""
    wave, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != cfg.SAMPLE_RATE:
        raise ValueError(f"Expected {cfg.SAMPLE_RATE} Hz, got {sr}: {path}")
    if wave.ndim != 1:
        raise ValueError(f"Expected mono, got shape {wave.shape}: {path}")

    n_samp    = cfg.N_SAMPLES
    hop_sec   = float(cfg.TTA_HOP_SEC)
    hop_samp  = int(round(hop_sec * cfg.SAMPLE_RATE))
    file_samp = int(round(cfg.FILE_DURATION * cfg.SAMPLE_RATE))
    n_windows = max(1, int((file_samp - n_samp) // hop_samp) + 1)

    segs = np.zeros((n_windows, n_samp), dtype=np.float32)
    for k in range(n_windows):
        s = k * hop_samp
        e = min(s + n_samp, len(wave))
        if e > s:
            segs[k, : e - s] = wave[s:e]

    out_chunks = []
    for i in range(0, n_windows, batch_size):
        batch = torch.from_numpy(segs[i:i + batch_size]).to(
            device, non_blocking=True
        )
        mel = mel_extractor(batch)
        logits = model(mel)
        out_chunks.append(torch.sigmoid(logits).float().cpu().numpy())
    tta_probs = np.concatenate(out_chunks, axis=0)

    probs = _aggregate_tta_to_official(
        tta_probs, hop_sec=hop_sec, win_sec=float(cfg.DURATION),
        n_output=cfg.N_OUTPUT_WIN, agg=cfg.TTA_AGG,
    )
    return probs, path.stem


def _load_submit_model(cfg, ckpt_path, device):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = BirdCLEFModel(cfg, with_proj=False).to(device).eval()
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=False,
    )
    drop_proj = [k for k in unexpected if k.startswith("proj_head.")]
    other_unexpected = [k for k in unexpected if not k.startswith("proj_head.")]
    if missing or other_unexpected:
        raise SystemExit(
            f"State_dict mismatch. missing={missing} unexpected={other_unexpected}"
        )
    if drop_proj:
        print(f"  ignored {len(drop_proj)} proj_head.* keys")
    print(f"  loaded (val_auc={ckpt.get('val_auc', float('nan')):.4f}, "
          f"epoch={ckpt.get('epoch', '?')})")
    return model


def _rank_normalize(probs):
    return pd.DataFrame(probs).rank(axis=0).to_numpy() / len(probs)


def submit(cfg):
    """Run inference, rank-average across checkpoints, optionally apply
    the post-processing chain, write submission.csv."""
    cfg.PRETRAINED = False
    device = torch.device(cfg.DEVICE)
    paths = list(cfg.SUBMIT_CKPT_PATHS)
    if not paths:
        raise SystemExit("CFG.SUBMIT_CKPT_PATHS is empty.")
    print(f"Submit mode | device={device} | ensemble of {len(paths)} models")

    test_dir = cfg.KAGGLE_DATA_DIR / "test_soundscapes"
    files = sorted(test_dir.glob("*.ogg"))
    if files:
        source = str(test_dir)
    elif cfg.SUBMIT_FALLBACK_FILES > 0:
        fallback_dir = cfg.KAGGLE_DATA_DIR / "train_soundscapes"
        files = sorted(fallback_dir.glob("*.ogg"))[:cfg.SUBMIT_FALLBACK_FILES]
        if not files:
            raise SystemExit(
                f"No OGG files found in {test_dir} or {fallback_dir}"
            )
        source = f"{fallback_dir} (save-and-run fallback)"
    else:
        raise SystemExit(f"No OGG files found in {test_dir}")
    print(f"Inference set: {len(files)} files from {source}")

    sample_cols = pd.read_csv(
        cfg.KAGGLE_DATA_DIR / "sample_submission.csv", nrows=0,
    ).columns.tolist()
    submission_cols = sample_cols[1:]
    labels_list, label_to_idx = get_label_map(
        cfg.KAGGLE_DATA_DIR / "taxonomy.csv",
    )
    if submission_cols != labels_list:
        raise SystemExit(
            "Column-order mismatch between sample_submission and taxonomy."
        )

    mel_extractor = LogMelExtractor(cfg).to(device).eval()
    n_classes = len(submission_cols)

    if cfg.TTA_ENABLED:
        print(f"TTA: on  hop_sec={cfg.TTA_HOP_SEC}  agg={cfg.TTA_AGG}")
        predict_fn = _predict_one_file_tta
    else:
        print("TTA: off")
        predict_fn = _predict_one_file

    per_model_probs = []
    rows = None
    for mi, ckpt_path in enumerate(paths):
        model = _load_submit_model(cfg, ckpt_path, device)
        m_rows, m_probs = [], []
        t0 = time.time()
        for path in tqdm(files, desc=f"infer[{mi + 1}/{len(paths)}]"):
            probs, stem = predict_fn(
                cfg, path, model, mel_extractor, device,
                cfg.SUBMIT_INFER_BATCH,
            )
            m_probs.append(probs)
            m_rows.extend(f"{stem}_{end}" for end in range(5, 65, 5))
        elapsed = time.time() - t0
        print(f"  model {mi + 1}: {len(files)} files in {elapsed:.1f}s "
              f"(avg {elapsed / max(1, len(files)):.3f}s/file)")

        if rows is None:
            rows = m_rows
        elif m_rows != rows:
            raise SystemExit(
                f"Row-id mismatch between model {mi + 1} and previous."
            )
        per_model_probs.append(np.concatenate(m_probs, axis=0))
        del model
        torch.cuda.empty_cache()
        gc.collect()

    if len(per_model_probs) == 1:
        P = per_model_probs[0]
        print("Single model - no ensemble needed.")
    else:
        ranks = [_rank_normalize(p) for p in per_model_probs]
        P = np.mean(ranks, axis=0)
        print(f"Ensembled {len(ranks)} models via per-class rank averaging.")

    if cfg.POSTPROC_ENABLED:
        P = apply_postproc_chain(cfg, P, rows, label_to_idx)
    else:
        print("Post-processing: off")

    df = pd.DataFrame(P, columns=submission_cols)
    df.insert(0, "row_id", rows)
    df.to_csv(cfg.SUBMIT_OUT_CSV, index=False)
    print(f"Wrote {cfg.SUBMIT_OUT_CSV} "
          f"({len(df):,} rows x {n_classes} species)")
