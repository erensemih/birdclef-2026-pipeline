"""Training and validation loops, plus run_holdout and the train() entry."""
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from data import (
    BirdDataset,
    SoundscapeCropDataset,
    build_focal_df,
    build_soundscape_df,
    get_label_map,
)
from mixup import maybe_mixup
from model import (
    BirdCLEFModel,
    LogMelExtractor,
    competition_macro_auc,
    seed_everything,
)
from perch import PerchTeacher
from prepare_data import ensure_wav_dirs


def train_one_epoch(cfg, model, mel_extractor, teacher, loader,
                    optimizer, scheduler, scaler, criterion, device):
    model.train()
    losses, cls_losses, kd_losses = [], [], []
    n_mixed = n_steps = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for wave, target in pbar:
        wave   = wave.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        wave, target, mixed = maybe_mixup(cfg, wave, target)
        n_steps += 1
        n_mixed += int(mixed)

        perch_emb = teacher.embed(wave) if teacher is not None else None
        mel = mel_extractor(wave)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=cfg.DEVICE, dtype=torch.bfloat16,
                      enabled=cfg.USE_AMP):
            if teacher is not None:
                logits, proj = model(mel, return_proj=True)
                cls_loss = criterion(logits, target)
                kd_loss = 1.0 - F.cosine_similarity(
                    proj.float(), perch_emb.float(), dim=-1
                ).mean()
                loss = cls_loss + cfg.KD_LAMBDA * kd_loss
            else:
                logits = model(mel)
                cls_loss = criterion(logits, target)
                kd_loss = None
                loss = cls_loss

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite training loss; aborting. cls={cls_loss.item():.4f}"
            )

        if cfg.USE_AMP:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            stepped = scaler.get_scale() >= scale_before
        else:
            loss.backward()
            optimizer.step()
            stepped = True
        if stepped:
            scheduler.step()

        losses.append(loss.item())
        cls_losses.append(cls_loss.item())
        postfix = {
            "loss": f"{np.mean(losses[-50:]):.4f}",
            "cls":  f"{np.mean(cls_losses[-50:]):.3f}",
            "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
        }
        if kd_loss is not None:
            kd_losses.append(kd_loss.item())
            postfix["kd"] = f"{np.mean(kd_losses[-50:]):.3f}"
        if cfg.MIXUP_ENABLED:
            postfix["mix"] = f"{n_mixed / max(1, n_steps):.2f}"
        pbar.set_postfix(**postfix)

    return float(np.mean(losses))


@torch.no_grad()
def validate(cfg, model, mel_extractor, loader, criterion, device):
    model.eval()
    all_probs, all_targets, losses = [], [], []
    for wave, target in tqdm(loader, desc="val", leave=False):
        wave   = wave.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mel = mel_extractor(wave)
        with autocast(device_type=cfg.DEVICE, dtype=torch.bfloat16,
                      enabled=cfg.USE_AMP):
            logits = model(mel)
            loss = criterion(logits, target)
        losses.append(loss.item())
        all_probs.append(torch.sigmoid(logits).float().cpu().numpy())
        all_targets.append(target.cpu().numpy())

    probs   = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    macro_auc, n_eval = competition_macro_auc(targets, probs)
    return float(np.mean(losses)), macro_auc, n_eval, probs, targets


def run_holdout(cfg, train_df, train_labels, val_df, val_labels):
    print("\n" + "=" * 60)
    print(" Training (train=focal+soundscape, val=soundscape, LEAKY)")
    print("=" * 60)
    print(f"Train: {len(train_df):,} rows (focal + soundscape segments)")
    print(f"Val:   {len(val_df):,} rows (soundscape, also in train)")
    print(f"Soft secondary coeff: {cfg.SOFT_LABEL_COEFF:.2f}"
          + (" (off)" if cfg.SOFT_LABEL_COEFF == 0.0 else ""))
    print(f"KD: {'on' if cfg.KD_ENABLED else 'off'}"
          + (f"  lambda={cfg.KD_LAMBDA:.2f}" if cfg.KD_ENABLED else ""))
    if cfg.MIXUP_ENABLED:
        print(f"Mixup: {cfg.MIXUP_MODE}  prob={cfg.MIXUP_PROB}")
    else:
        print("Mixup: off")

    train_ds = BirdDataset(cfg, train_df, train_labels, train=True)
    if cfg.SS_CROP_MODE != "off":
        _, label_to_idx = get_label_map(cfg.TAXONOMY_CSV)
        ss_crop_ds = SoundscapeCropDataset(cfg, label_to_idx)
        print(f"Extra SS crops ({cfg.SS_CROP_MODE}): "
              f"+{len(ss_crop_ds):,} samples/epoch")
        train_ds = ConcatDataset([train_ds, ss_crop_ds])

    val_ds = BirdDataset(cfg, val_df, val_labels, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )

    seed_everything(cfg.SEED)
    mel_extractor = LogMelExtractor(cfg).to(cfg.DEVICE).eval()
    model = BirdCLEFModel(cfg, with_proj=cfg.KD_ENABLED).to(cfg.DEVICE)

    teacher = None
    if cfg.KD_ENABLED:
        teacher = PerchTeacher(
            cfg.PERCH_PATH, torch.device(cfg.DEVICE),
            expected_embed_dim=cfg.PERCH_EMBED_DIM,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
    )
    # OneCycleLR needs pct_start in (0, 1). Clamp so sanity runs with
    # EPOCHS <= WARMUP_EPOCHS don't blow up the scheduler.
    pct_start = min(0.5, max(1e-3, cfg.WARMUP_EPOCHS / cfg.EPOCHS))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.LR, epochs=cfg.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=pct_start,
        div_factor=cfg.DIV_FACTOR, final_div_factor=cfg.FINAL_DIV_FACTOR,
        anneal_strategy="cos",
    )
    hard_pos = (train_labels >= 1.0 - 1e-6)
    n_pos = hard_pos.sum(axis=0).astype(np.float64)
    n_neg = train_labels.shape[0] - n_pos
    pos_weight_np = np.where(n_pos > 0, n_neg / np.maximum(n_pos, 1.0), 1.0)
    pos_weight_np = np.clip(pos_weight_np, 1.0, cfg.POS_WEIGHT_CLIP)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=cfg.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler    = GradScaler(device=cfg.DEVICE, enabled=False)

    best_auc, best_probs, best_epoch = 0.0, None, -1
    for epoch in range(cfg.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{cfg.EPOCHS} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} ---")
        tr_loss = train_one_epoch(
            cfg, model, mel_extractor, teacher, train_loader,
            optimizer, scheduler, scaler, criterion, cfg.DEVICE,
        )
        va_loss, va_auc, n_eval, va_probs, _ = validate(
            cfg, model, mel_extractor, val_loader, criterion, cfg.DEVICE,
        )
        print(f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"val_macro_auc={va_auc:.4f}  ({n_eval} classes)")

        if va_auc > best_auc:
            best_auc, best_probs, best_epoch = va_auc, va_probs, epoch
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_auc": va_auc,
            }, cfg.TRAIN_CKPT_PATH)
            print(f"  -> new best (auc={va_auc:.4f}); saved {cfg.TRAIN_CKPT_PATH}")

    torch.save({
        "epoch": cfg.EPOCHS - 1, "model_state_dict": model.state_dict(),
        "val_auc": va_auc,
    }, cfg.LAST_CKPT_PATH)
    print(f"\nBest: AUC={best_auc:.4f} at epoch {best_epoch + 1}")
    print(f"Saved last -> {cfg.LAST_CKPT_PATH}")

    del model, mel_extractor, optimizer, scheduler, scaler
    del train_loader, val_loader, train_ds, val_ds
    if teacher is not None:
        del teacher
    torch.cuda.empty_cache()
    gc.collect()

    return best_auc, best_probs, best_epoch


def train(cfg):
    """Build the full train_df (focal + soundscape, LEAKY) and run a
    single training pass."""
    seed_everything(cfg.SEED)
    cfg.TRAIN_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ensure_wav_dirs(cfg)

    labels_list, label_to_idx = get_label_map(cfg.TAXONOMY_CSV)
    print(f"Number of classes: {len(labels_list)}")

    focal_df, focal_labels = build_focal_df(cfg, label_to_idx)
    ss_df,    ss_labels    = build_soundscape_df(cfg, label_to_idx)
    print(f"Focal recordings:    {len(focal_df):,}")
    print(f"Soundscape segments: {len(ss_df):,} from "
          f"{ss_df['group'].nunique()} files")

    train_df     = pd.concat([focal_df, ss_df], axis=0, ignore_index=True)
    train_labels = np.concatenate([focal_labels, ss_labels], axis=0)
    val_df       = ss_df.reset_index(drop=True)
    val_labels   = ss_labels.copy()

    best_auc, _, best_epoch = run_holdout(
        cfg, train_df, train_labels, val_df, val_labels,
    )

    print("\n" + "=" * 60)
    print(" Done")
    print("=" * 60)
    print(f"  Best val AUC (LEAKY): {best_auc:.4f}  (epoch {best_epoch + 1})")
    print(f"  Best ckpt: {cfg.TRAIN_CKPT_PATH.resolve()}")
    print(f"  Last ckpt: {cfg.LAST_CKPT_PATH.resolve()}")
