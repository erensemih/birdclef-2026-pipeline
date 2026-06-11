"""Two mixup modes: classic convex, and temporal-concat."""
import numpy as np
import torch


def _mixup_convex(cfg, wave, target):
    """λ ~ Beta(α, α); wave/target ← lerp(wave, wave[π], λ)."""
    lam = float(np.random.beta(cfg.MIXUP_ALPHA, cfg.MIXUP_ALPHA))
    perm = torch.randperm(wave.size(0), device=wave.device)
    wave_out   = lam * wave   + (1.0 - lam) * wave[perm]
    target_out = lam * target + (1.0 - lam) * target[perm]
    return wave_out, target_out


def _mixup_temporal_concat(cfg, wave, target):
    """Split the 5-sec window into K contiguous chunks from K different
    batch samples. Equal-power sin/cos crossfade at each junction. Label
    is the duration-weighted sum of the full target vectors, clipped to
    [0, 1]. Falls back to convex when K < 2 or batch size is too small."""
    B = wave.size(0)
    N = wave.size(1)
    device = wave.device

    K = int(cfg.MIXUP_CONCAT_PIECES)
    if K < 2 or B < 2:
        return _mixup_convex(cfg, wave, target)

    min_samp = int(round(cfg.MIXUP_CONCAT_MIN_SEC * cfg.SAMPLE_RATE))
    min_samp = max(1, min(min_samp, N // K))
    xf = int(round(cfg.MIXUP_CONCAT_XFADE_SEC * cfg.SAMPLE_RATE))

    free = N - K * min_samp
    if free < 0:
        return _mixup_convex(cfg, wave, target)
    parts = (np.random.multinomial(free, [1.0 / K] * K)
             if free > 0 else np.zeros(K, dtype=np.int64))
    seg_len = (parts + min_samp).astype(np.int64)
    seg_len[-1] += N - int(seg_len.sum())
    bounds = np.concatenate([[0], np.cumsum(seg_len)]).astype(np.int64)

    perms = [torch.arange(B, device=device)]
    for _ in range(1, K):
        perms.append(torch.randperm(B, device=device))

    if xf > 0:
        t = torch.arange(xf, device=device, dtype=wave.dtype)
        fade_in  = torch.sin(0.5 * np.pi * (t + 0.5) / xf)
        fade_out = torch.cos(0.5 * np.pi * (t + 0.5) / xf)

    wave_out = torch.empty_like(wave)
    for j in range(K):
        s, e = int(bounds[j]), int(bounds[j + 1])
        src = wave[perms[j]]
        wave_out[:, s:e] = src[:, s:e]
        if xf > 0:
            xj = min(xf, (e - s) // 2 if (e - s) >= 2 else 0)
            if xj > 0 and j > 0:
                wave_out[:, s:s + xj] = src[:, s:s + xj] * fade_in[:xj]
            if xj > 0 and j < K - 1:
                wave_out[:, e - xj:e] = src[:, e - xj:e] * fade_out[:xj]

    w = torch.tensor(seg_len, device=device, dtype=target.dtype) / float(N)
    target_out = torch.zeros_like(target)
    for j in range(K):
        target_out = target_out + w[j] * target[perms[j]]
    target_out = target_out.clamp_(0.0, 1.0)
    return wave_out, target_out


def maybe_mixup(cfg, wave, target):
    """Apply mixup with probability MIXUP_PROB. Returns (wave, target, mixed)."""
    if not cfg.MIXUP_ENABLED or np.random.random() >= cfg.MIXUP_PROB:
        return wave, target, False
    if cfg.MIXUP_MODE == "temporal_concat":
        out = _mixup_temporal_concat(cfg, wave, target)
    else:
        out = _mixup_convex(cfg, wave, target)
    return out[0], out[1], True
