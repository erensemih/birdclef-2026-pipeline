# BirdCLEF+ 2026 — pipeline notes

The audio-tagging pipeline put together for the
[BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)
competition. Nothing exotic: a CNN student, Perch v2 as a frozen
teacher, a few inference-time tricks, and a couple of training-time
augmentations that did not end up helping. The submissions did not
place well; this repository exists mainly to keep the iteration in
one place.

The shared code lives in a few small modules. Each numbered script is
a thin `CFG` subclass that flips a couple of flags and calls `train()`
or `submit()`.

## Scripts

```
01_baseline.py               KD-on baseline, no mixup, no augment
02_mixup.py                  + classic 2-sample mixup
03_inference_postproc.py     + submit-time post-processing (4 stages)
04_tta_final.py              + multi-crop TTA at submit   (final config)
ablation_ss_augment.py       random SS crop augmentation  (no LB gain)
ablation_temporal_concat.py  dense SS crops + temporal-concat mixup (no LB gain)
```

## Modules

```
cfg.py        BaseCFG: paths, audio, model, training, KD, mixup, SS-crop,
              submit, TTA, post-processing — all overridable per script.
data.py       Audio I/O, label maps, parse_fname, BirdDataset,
              SoundscapeCropDataset (random or dense mode).
model.py      LogMelExtractor, EfficientNet-B0 backbone with optional
              KD projection head, AUC, seeding.
perch.py      Perch v2 SavedModel wrapper used for online KD.
mixup.py      Classic convex mixup, temporal-concat mixup, dispatcher.
loop.py       train_one_epoch, validate, run_holdout, train() entry.
infer.py      Single-window predict, TTA predict, submit() entry
              with rank-average ensemble.
postproc.py   Site/hour priors, t-smoothing, sonotype mirroring,
              rare-class suppression, and the chain that runs them.
```

## What is shared by every script

### Backbone

EfficientNet-B0 via `timm`, single-channel input, average-pool head,
234-way classifier. A separate linear projection head is added when
KD is enabled (output dim = Perch's 1536); the projection head is
used only at training time and dropped at submit.

### Audio → mel

5-second crops at 32 kHz, log-mel spectrogram with 256 mel bins.
`WIN_LENGTH` and `HOP_LENGTH` are chosen so the time dimension lands
on 256 natively (no resize). The dB output is clamped to a fixed
`[-TOP_DB, 0]` range and linearly mapped to `[0, 1]`. `f_min=20 Hz`
keeps frog and mammal calls audible.

### KD from Perch v2

Perch v2 is loaded as a frozen TensorFlow SavedModel. Each training
batch's waveform is sent through Perch on the same GPU and the
1536-d `embedding` output is used as the KD target. The student's
projection head is pulled toward it by a cosine loss:

```
loss = BCE(logits, target) + λ_KD · (1 − cos(student_proj, perch_emb))
```

KD applies to every training row, focal or soundscape.

### Soft secondary labels

Focal recordings carry a primary species (label = 1.0) and any
secondary species at a soft value (default 0.3). Soundscape rows are
hard multi-hot.

### Leaky full-train

The labelled train soundscape segments are put into the training set
as themselves, and the same segments are also the validation set, so
`val_macro_auc` here is leaky. Only the trend between epochs was
used as the stop signal, not the absolute value.

### Submit ensemble

Each training run writes two checkpoints: the best-by-val-AUC one and
the final-epoch one. At submit time both are loaded, inference is run
independently per checkpoint, and the two probability matrices are
combined with per-class rank averaging.

## What each script adds

### `01_baseline.py`

Just the shared pieces above. Useful as the starting point that
nothing else removes.

### `02_mixup.py`

Adds classic 2-sample mixup on the training batch:

```
λ ~ Beta(α, α)
π = random permutation of the batch
wave   ← λ·wave   + (1−λ)·wave[π]
target ← λ·target + (1−λ)·target[π]
```

The Perch teacher runs on the mixed waveform so the KD target
matches the student's input.

### `03_inference_postproc.py`

Adds four submit-time post-processing stages applied after the
ensemble, before writing `submission.csv`:

1. **Site/hour priors.** Filenames carry site + UTC hour
   (`BC2026_<Train|Test>_<id>_<S##>_<date>_<HHMMSS>.ogg`).
   Per-species priors are estimated at four granularities (global,
   hour, site, site×hour) with adaptive shrinkage when a fine cell
   has too few samples. Each row gets a logit-shift toward its
   prior:  `logits += λ · ( log p − log(1 − p) )`.
2. **Temporal smoothing.** Across the 12 windows of each test file,
   probabilities are smoothed with a t-distribution kernel
   (radius 3, df = 1.2) and blended back with the raw values by
   `α = 0.3`.
3. **Sonotype mirroring.** A few labelled `47158son*` insect codes
   are basically the same call; group members share the per-row
   max.
4. **Rare-class suppression.** Predictions in `Amphibia`,
   `Mammalia` and `Reptilia` rows below `column_mean + RARE_MARGIN`
   get scaled down by `RARE_SCALE`. Keeps spurious rare-class
   blooms in check.

Each stage has its own enable flag; the whole chain is also gated
behind `CFG.POSTPROC_ENABLED`.

### `04_tta_final.py`

Adds multi-crop TTA in the submit prediction loop. Each 60-sec test
clip is sliced at 5-sec windows with a `TTA_HOP_SEC` (default 2.5)
stride, giving 23 overlapping windows instead of the 12 default.
They are aggregated back to the 12 official output positions by an
overlap-area-weighted mean. This is the configuration used for the
final submission.

## Ablations — did not help

### `ablation_ss_augment.py`

Random 5-sec crops drawn anywhere inside each labelled train
soundscape's span are added to the training set with overlap-weighted
multi-hot labels (Option A):

```
label[s] = clip( Σ_seg overlap_sec(crop, seg) / 5 , 0, 1 )   s ∈ seg
```

The hope was that more multi-species 5-sec windows that look like
the test format would help. It did not. Kept here for the record.

### `ablation_temporal_concat.py`

Two more changes on top of the SS crops:

1. **Dense SS crops.** Random draws replaced by deterministic
   enumeration at a fixed stride. With 66 labelled files this caps
   at about 3.5k unique crops per epoch at a 1-second stride.
2. **Temporal-concat mixup.** Instead of overlaying two clips, the
   5-sec window is split into K contiguous chunks summing to 5 sec,
   each filled from a different batch sample. Equal-power sin/cos
   crossfade at the junctions removes the splice click. The label
   is the duration-weighted sum of the full target vectors. The
   motivation was that two real clips concatenated end-to-end sound
   more like a real ambient recording than two clips overlaid.

Neither change moved the leaderboard.

## Data layout

```
birdclef-2026/                 original Kaggle data (OGG)
  train_audio/                 focal recordings as Kaggle ships them
  train_soundscapes/           labelled soundscapes as Kaggle ships them
birdclef-2026-wav/             local 32 kHz mono WAV copy (auto-created)
  train_audio_wav/
  train_soundscapes_wav/
taxonomy.csv                   234-class taxonomy
train.csv                      focal labels
train_soundscapes_labels.csv   5-sec multi-label segments
perch_v2/                      Perch v2 SavedModel directory
```

Training reads from the WAV directories (random-access seek into mid-
file crops is much faster on WAV than on Vorbis). If those WAV folders
are empty on first run, `prepare_data.ensure_wav_dirs(cfg)` is called
automatically from `train()` and converts the matching OGG files to
32 kHz mono PCM, mirroring the directory structure. The conversion
can also be triggered manually:

```bash
cd repo
python prepare_data.py
```

The Kaggle submit path is the OGG `test_soundscapes` directory under
`/kaggle/input/competitions/birdclef-2026/`; no conversion needed
there.

## Usage

```bash
pip install -r requirements.txt
```

Each script picks its mode from `CFG.MODE`. Edit the field inside
the file:

```python
CFG.MODE = "train"   # local; writes best_model_<name>.pt + last_model_<name>.pt
CFG.MODE = "submit"  # Kaggle; loads CFG.SUBMIT_CKPT_PATHS, writes submission.csv
```

Then:

```bash
cd repo
python 04_tta_final.py
```

For Kaggle submit, upload the trained checkpoints as a Kaggle
dataset matching the layout in `CFG.SUBMIT_CKPT_PATHS`, set
`CFG.MODE = "submit"`, and run the script in the notebook.

## Notes

- Only the `_full` (leaky) training variant is included here. The
  hold-out variants (focal-only train, soundscape val) were used
  during exploration but are not in this repo.
- A few exploratory variants — a SED attention head, an
  `eca_nfnet_l0` backbone, a LinearLR schedule, an ICON-style
  contrastive pretext, and others — were tried and not adopted;
  they are also not in this repo.
- Paths and a few magic numbers are sprinkled across `cfg.py` and
  the per-script `CFG` subclasses. Edit before running.

## License

[MIT](LICENSE).
