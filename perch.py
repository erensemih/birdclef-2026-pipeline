"""Frozen Perch v2 SavedModel wrapper for online KD."""
from pathlib import Path

import numpy as np
import torch


class PerchTeacher:
    """Loads Perch v2 from a TF SavedModel and exposes embed()."""

    def __init__(self, model_path, device, expected_embed_dim):
        try:
            import tensorflow as tf  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "Online Perch distillation needs TensorFlow. Install it, "
                "or set CFG.KD_ENABLED = False."
            ) from e

        if not Path(model_path).exists():
            raise SystemExit(f"Perch SavedModel not found at {model_path}.")

        for g in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError:
                pass

        self.tf = tf
        print(f"Loading Perch v2 SavedModel: {model_path}")
        self.module = tf.saved_model.load(str(model_path))
        if "serving_default" not in self.module.signatures:
            raise SystemExit(
                "Perch SavedModel has no 'serving_default' signature."
            )
        self.signature = self.module.signatures["serving_default"]
        out_keys = list(self.signature.structured_outputs.keys())
        if "embedding" not in out_keys:
            raise SystemExit(
                f"Perch needs 'embedding' output; got: {out_keys}"
            )
        self.device = device
        self.embed_dim = expected_embed_dim
        print(f"Perch ready; signature outputs: {out_keys}")

    def embed(self, wave):
        wave_np = wave.detach().cpu().numpy().astype(np.float32, copy=False)
        out = self.signature(inputs=self.tf.constant(wave_np))
        emb = out["embedding"].numpy()
        return torch.from_numpy(emb).to(self.device, non_blocking=True)
