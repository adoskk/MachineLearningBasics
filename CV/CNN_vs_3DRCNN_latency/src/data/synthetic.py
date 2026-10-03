"""Synthetic CT fallback — lets you smoke-test everything with NO HF access.

Generates volumes with 1-3 bright "lesion" spheres on noisy background,
plus exact boxes + slice labels. Used when:
  - use_synthetic: true in config, or
  - CADS download not available (CPU dev on Windows, CI).
"""
from __future__ import annotations

import numpy as np

from .cads import VolumeSample, mask_to_boxes, mask_to_slice_labels


def make_synthetic_volume(
    shape=(64, 128, 128),
    n_lesions: int = 2,
    rng: np.random.Generator | None = None,
    volume_id: str = "synth_000",
) -> VolumeSample:
    rng = rng or np.random.default_rng(0)
    D, H, W = shape
    vol = rng.normal(0.35, 0.08, size=shape).astype(np.float32)  # soft-tissue-ish bg
    mask = np.zeros(shape, dtype=np.uint8)
    zz, yy, xx = np.ogrid[:D, :H, :W]
    for _ in range(n_lesions):
        cz, cy, cx = (int(rng.integers(8, D - 8)), int(rng.integers(20, H - 20)), int(rng.integers(20, W - 20)))
        r = int(rng.integers(6, 14))
        sphere = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        mask[sphere] = 1
        vol[sphere] = np.clip(vol[sphere] + rng.uniform(0.25, 0.45), 0, 1)
    vol = np.clip(vol, 0, 1).astype(np.float32)
    boxes = mask_to_boxes(mask, min_voxels=50, min_size=3)
    sl = mask_to_slice_labels(mask, thresh=10)
    return VolumeSample(volume_id=volume_id, volume=vol, mask=mask, boxes=boxes, slice_labels=sl)


def make_synthetic_cohort(n: int = 8, shape=(64, 128, 128), seed: int = 0):
    rng = np.random.default_rng(seed)
    return [
        make_synthetic_volume(shape, n_lesions=int(rng.integers(1, 4)), rng=rng, volume_id=f"synth_{i:03d}")
        for i in range(n)
    ]
