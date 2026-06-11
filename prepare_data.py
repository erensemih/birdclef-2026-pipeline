"""Convert source OGG files to 32 kHz mono WAV on first use.

Why WAV: soundfile + .wav allows partial reads (seek + read N frames),
which makes per-epoch loading noticeably faster than decoding .ogg
every time.

ensure_wav_dirs(cfg) is a no-op when the target WAV directories already
contain WAVs. Otherwise it walks the matching OGG layout
(CFG.OGG_TRAIN_AUDIO_DIR / CFG.OGG_SOUNDSCAPE_DIR), resamples each file
to CFG.SAMPLE_RATE mono, and mirrors the directory structure. Per-file
skip means re-running is safe — partially converted dirs just resume.

train() in loop.py calls this at startup, so the conversion happens
automatically the first time training is run. Standalone use:

    python prepare_data.py             # convert whatever is missing
    python prepare_data.py --all       # force-walk both dirs
    python prepare_data.py --train-audio
    python prepare_data.py --soundscapes

Disk space (FLOAT WAV, 32 kHz mono):
  train_audio:        ~90 GB    (PCM_16: ~45 GB)
  train_soundscapes:  ~80 GB    (PCM_16: ~40 GB)
"""
import argparse
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm


WAV_SUBTYPE = "FLOAT"   # "FLOAT" (32-bit) or "PCM_16" (half size, ~no quality loss)


def _load_audio_mono(path, target_sr):
    """Load any audio file as mono float32 at target_sr. Falls back to
    librosa when soundfile cannot decode the source."""
    try:
        wave, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        wave, sr = librosa.load(str(path), sr=None, mono=False)

    if wave.ndim > 1:
        # soundfile returns (samples, channels); librosa returns (channels, samples)
        axis = 1 if wave.shape[0] > wave.shape[1] else 0
        wave = wave.mean(axis=axis)

    if sr != target_sr:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=target_sr)

    return wave.astype(np.float32)


def _convert_one_file(src, dst, target_sr):
    """Convert one OGG -> WAV. Returns 'ok' / 'skipped' / 'failed: ...'."""
    if dst.exists():
        return "skipped"
    try:
        wave = _load_audio_mono(src, target_sr)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), wave, samplerate=target_sr,
                 subtype=WAV_SUBTYPE, format="WAV")
        return "ok"
    except Exception as e:
        return f"failed: {e}"


def _has_wavs(d):
    if not Path(d).exists():
        return False
    for _ in Path(d).rglob("*.wav"):
        return True
    return False


def _build_tasks(src_dir, dst_dir):
    """Walk src_dir for *.ogg, mirror the tree under dst_dir as *.wav."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    tasks = []
    for s in sorted(src_dir.rglob("*.ogg")):
        rel = s.relative_to(src_dir)
        d = dst_dir / rel.with_suffix(".wav")
        tasks.append((s, d))
    return tasks


def convert_dir(src_dir, dst_dir, target_sr, n_jobs, label):
    src_dir = Path(src_dir)
    if not src_dir.exists():
        raise SystemExit(
            f"Source OGG directory not found: {src_dir}\n"
            f"  Download the Kaggle competition data so this folder exists, "
            f"or update CFG.OGG_BASE_DIR to point at the OGG layout."
        )
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    tasks = _build_tasks(src_dir, dst_dir)
    print(f"\n=== {label} ===")
    print(f"Found {len(tasks):,} .ogg files under {src_dir}")
    if not tasks:
        return

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_convert_one_file)(s, d, target_sr)
        for s, d in tqdm(tasks, desc=label)
    )
    summary = Counter(
        r if r in ("ok", "skipped") else "failed" for r in results
    )
    print(f"Done {label}. Summary: {dict(summary)}")
    failures = [r for r in results if r.startswith("failed")]
    if failures:
        print(f"First 5 failures: {failures[:5]}")


def ensure_wav_dirs(cfg):
    """If the WAV directories don't already contain WAVs, convert from
    the matching OGG layout. Safe to call at every train() startup."""
    todo = []
    if not _has_wavs(cfg.TRAIN_AUDIO_DIR):
        todo.append((cfg.OGG_TRAIN_AUDIO_DIR, cfg.TRAIN_AUDIO_DIR,
                     "train_audio"))
    if not _has_wavs(cfg.SOUNDSCAPE_DIR):
        todo.append((cfg.OGG_SOUNDSCAPE_DIR, cfg.SOUNDSCAPE_DIR,
                     "train_soundscapes"))
    if not todo:
        return

    print("=" * 60)
    print(" Preparing audio (OGG -> WAV)")
    print("=" * 60)
    print(f"Format:  WAV {WAV_SUBTYPE} @ {cfg.SAMPLE_RATE} Hz mono")
    n_jobs = max(1, int(getattr(cfg, "NUM_WORKERS", 8)))
    print(f"Workers: {n_jobs}")
    for src, dst, name in todo:
        convert_dir(src, dst, cfg.SAMPLE_RATE, n_jobs=n_jobs, label=name)
    print("Done preparing audio.\n")


def _main():
    from cfg import BaseCFG

    p = argparse.ArgumentParser()
    p.add_argument("--train-audio", action="store_true",
                   help="force-walk train_audio/")
    p.add_argument("--soundscapes", action="store_true",
                   help="force-walk train_soundscapes/")
    p.add_argument("--all", action="store_true",
                   help="force-walk both")
    args = p.parse_args()

    if not (args.all or args.train_audio or args.soundscapes):
        # Default: skip dirs that already have WAVs
        ensure_wav_dirs(BaseCFG)
        return

    n_jobs = max(1, int(BaseCFG.NUM_WORKERS))
    print(f"Source:  {BaseCFG.OGG_BASE_DIR}")
    print(f"Output:  {BaseCFG.WAV_DIR}")
    print(f"Format:  WAV {WAV_SUBTYPE} @ {BaseCFG.SAMPLE_RATE} Hz mono")
    print(f"Workers: {n_jobs}")

    if args.all or args.train_audio:
        convert_dir(BaseCFG.OGG_TRAIN_AUDIO_DIR, BaseCFG.TRAIN_AUDIO_DIR,
                    BaseCFG.SAMPLE_RATE, n_jobs=n_jobs, label="train_audio")
    if args.all or args.soundscapes:
        convert_dir(BaseCFG.OGG_SOUNDSCAPE_DIR, BaseCFG.SOUNDSCAPE_DIR,
                    BaseCFG.SAMPLE_RATE, n_jobs=n_jobs, label="train_soundscapes")


if __name__ == "__main__":
    _main()
