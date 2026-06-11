"""Submit-time post-processing: site/hour priors, t-smoothing,
sonotype mirroring, rare-class suppression."""
import numpy as np
import pandas as pd

from data import get_class_name_map, parse_fname


def build_prior_tables(cfg, label_to_idx, num_classes):
    """Estimate per-species prior probabilities at four granularities
    (global, hour, site, site×hour) from the train soundscape labels."""
    df = pd.read_csv(cfg.KAGGLE_DATA_DIR / "train_soundscapes_labels.csv")
    df = df.drop_duplicates().reset_index(drop=True)

    Y = np.zeros((len(df), num_classes), dtype=np.float32)
    for i, label_str in enumerate(df["primary_label"].astype(str).values):
        for lab in label_str.split(";"):
            lab = lab.strip()
            if lab in label_to_idx:
                Y[i, label_to_idx[lab]] = 1.0

    meta = df["filename"].apply(parse_fname).apply(pd.Series)
    df = pd.concat([df, meta], axis=1)
    valid = (df["site"] != "unknown") & (df["hour_utc"] >= 0)
    df = df[valid].reset_index(drop=True)
    Y  = Y[valid.values]

    global_p = Y.mean(axis=0).astype(np.float32)

    sv = df["site"].astype(str).values
    site_keys = sorted(df["site"].astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p = np.zeros((len(site_keys), num_classes), dtype=np.float32)
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    for s in site_keys:
        mask = sv == s
        i = site_to_i[s]
        site_n[i] = mask.sum()
        site_p[i] = Y[mask].mean(axis=0)

    hv = df["hour_utc"].astype(int).values
    hour_keys = sorted(df["hour_utc"].astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_p = np.zeros((len(hour_keys), num_classes), dtype=np.float32)
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    for h in hour_keys:
        mask = hv == h
        i = hour_to_i[h]
        hour_n[i] = mask.sum()
        hour_p[i] = Y[mask].mean(axis=0)

    sh_keys = sorted({(str(s), int(h)) for s, h in zip(sv, hv)})
    sh_to_i = {k: i for i, k in enumerate(sh_keys)}
    sh_p = np.zeros((len(sh_keys), num_classes), dtype=np.float32)
    sh_n = np.zeros(len(sh_keys), dtype=np.float32)
    for s, h in sh_keys:
        mask = (sv == s) & (hv == h)
        i = sh_to_i[(s, h)]
        sh_n[i] = mask.sum()
        sh_p[i] = Y[mask].mean(axis=0)

    print(f"  priors: {len(site_keys)} sites, {len(hour_keys)} hours, "
          f"{len(sh_keys)} site x hour buckets  (n_rows={len(df):,})")
    return {
        "global_p":  global_p,
        "site_to_i": site_to_i, "site_p": site_p, "site_n": site_n,
        "hour_to_i": hour_to_i, "hour_p": hour_p, "hour_n": hour_n,
        "sh_to_i":   sh_to_i,   "sh_p":   sh_p,   "sh_n":   sh_n,
    }


def apply_prior(cfg, probs, sites, hours, tables):
    """Logit-shift each row by lambda * logit(prior). Per-row prior uses
    adaptive shrinkage stacked global -> hour -> site -> site x hour."""
    eps = 1e-5
    n = probs.shape[0]
    p_prior = np.tile(tables["global_p"], (n, 1)).astype(np.float32)
    k_h = cfg.PRIORS_HOUR_SHRINK
    k_s = cfg.PRIORS_SITE_SHRINK
    k_sh = cfg.PRIORS_SH_SHRINK

    for i in range(n):
        h = int(hours[i]); s = str(sites[i])
        hi = tables["hour_to_i"].get(h)
        if hi is not None:
            nh = tables["hour_n"][hi]
            w = nh / (nh + k_h)
            p_prior[i] = w * tables["hour_p"][hi] + (1 - w) * tables["global_p"]
        si = tables["site_to_i"].get(s)
        if si is not None:
            ns = tables["site_n"][si]
            w = ns / (ns + k_s)
            p_prior[i] = w * tables["site_p"][si] + (1 - w) * p_prior[i]
        shi = tables["sh_to_i"].get((s, h))
        if shi is not None:
            nsh = tables["sh_n"][shi]
            w = nsh / (nsh + k_sh)
            p_prior[i] = w * tables["sh_p"][shi] + (1 - w) * p_prior[i]

    p_prior = np.clip(p_prior, eps, 1 - eps)
    prior_logit = np.log(p_prior) - np.log1p(-p_prior)
    p = np.clip(probs, eps, 1 - eps)
    logits = np.log(p) - np.log1p(-p)
    logits = logits + cfg.PRIORS_LAMBDA * prior_logit
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _build_t_kernel(radius, df_param):
    offs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = (1.0 + (offs / df_param) ** 2 / 2.0) ** (-1.5)
    return (k / k.sum()).astype(np.float32)


def temporal_smooth(probs, file_ids, kernel, alpha):
    """Smooth predictions across the 12 windows of each file with a
    t-distribution kernel, then blend with raw via alpha."""
    radius = (len(kernel) - 1) // 2
    out = probs.copy()
    for fid in pd.unique(file_ids):
        mask = file_ids == fid
        x = probs[mask]
        if len(x) <= 1:
            continue
        padded = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
        smoothed = np.zeros_like(x)
        for i, w in enumerate(kernel):
            smoothed += w * padded[i:i + len(x)]
        out[mask] = (1.0 - alpha) * x + alpha * smoothed
    return out.astype(np.float32)


def sonotype_mirror(probs, label_to_idx, groups):
    """Within each labelled group, every member's prob becomes the
    per-row max over the group."""
    out = probs.copy()
    for group in groups:
        valid_idx = [label_to_idx[s] for s in group if s in label_to_idx]
        if len(valid_idx) < 2:
            continue
        group_max = out[:, valid_idx].max(axis=1, keepdims=True)
        for idx in valid_idx:
            out[:, idx] = group_max[:, 0]
    return out


def rare_class_suppress(probs, label_to_idx, class_name_map,
                        rare_class_names, margin, scale):
    """Scale down rare-class predictions that are below the column mean
    plus a margin."""
    out = probs.copy()
    rare_idxs = [
        label_to_idx[lab] for lab, cn in class_name_map.items()
        if cn in rare_class_names and lab in label_to_idx
    ]
    if not rare_idxs:
        return out
    for ci in rare_idxs:
        vals = out[:, ci]
        threshold = vals.mean() + margin
        out[:, ci] = np.where(vals < threshold, vals * scale, vals)
    return out


def apply_postproc_chain(cfg, P, rows, label_to_idx):
    """Run the four stages in order. Per-stage flags still apply."""
    file_ids = np.array(["_".join(r.split("_")[:-1]) for r in rows])
    meta = [parse_fname(fid + ".ogg") for fid in file_ids]
    sites = np.array([m["site"]     for m in meta])
    hours = np.array([m["hour_utc"] for m in meta], dtype=np.int64)
    n_parsed = int((sites != "unknown").sum())
    print(f"Site/hour parsed for {n_parsed}/{len(rows)} rows")

    if cfg.PRIORS_ENABLED and n_parsed > 0:
        print(f"Applying site-hour priors (lambda={cfg.PRIORS_LAMBDA})...")
        tables = build_prior_tables(cfg, label_to_idx, cfg.NUM_CLASSES)
        P = apply_prior(cfg, P, sites, hours, tables)
    else:
        print("Priors: off (or no parsable site/hour rows)")

    if cfg.TEMPSMOOTH_ENABLED:
        kernel = _build_t_kernel(cfg.TEMPSMOOTH_RADIUS, cfg.TEMPSMOOTH_DF)
        print(f"Applying temporal smoothing (radius={cfg.TEMPSMOOTH_RADIUS}, "
              f"df={cfg.TEMPSMOOTH_DF}, alpha={cfg.TEMPSMOOTH_ALPHA})...")
        P = temporal_smooth(P, file_ids, kernel, cfg.TEMPSMOOTH_ALPHA)
    else:
        print("Temporal smoothing: off")

    if cfg.SONOTYPE_GROUPS:
        print(f"Applying sonotype mirroring over "
              f"{len(cfg.SONOTYPE_GROUPS)} group(s)...")
        P = sonotype_mirror(P, label_to_idx, cfg.SONOTYPE_GROUPS)

    if cfg.RARE_SUPPRESS_ENABLED:
        class_name_map = get_class_name_map(cfg.KAGGLE_DATA_DIR / "taxonomy.csv")
        rare_classes = set(cfg.RARE_CLASS_NAMES)
        n_rare = sum(1 for cn in class_name_map.values() if cn in rare_classes)
        print(f"Applying rare-class suppression "
              f"(classes={cfg.RARE_CLASS_NAMES}, ~{n_rare} species)...")
        P = rare_class_suppress(
            P, label_to_idx, class_name_map, rare_classes,
            cfg.RARE_MARGIN, cfg.RARE_SCALE,
        )
    return P
