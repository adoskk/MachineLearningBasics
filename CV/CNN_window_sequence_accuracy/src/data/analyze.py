"""Exploratory data analysis for the lesion cohort (notebook §2 + standalone CLI).

Answers, for the exact volumes/splits training uses:
  - how many volumes per split (train/val/test)?
  - native resolution spread in x/y/z (grid shape) + voxel spacing spread?
  - target-object sizes (voxels + mm³) and objects-per-volume?
  - where do objects sit (normalized x/y/z centroids)?

Object stats are computed at NATIVE resolution (masks reloaded via
load_native_mask) so sizes are physically meaningful; resolution stats come
from NIfTI headers only (cheap, no voxel loads).

Outputs: console summary + `data_analysis_summary.json` + `eda_*.png` figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _components(mask: np.ndarray, spacing) -> list:
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask > 0)
    D, H, W = mask.shape
    rows = []
    for i in range(1, n + 1):
        zz, yy, xx = np.where(lab == i)
        v = len(zz)
        rows.append({
            "voxels": int(v),
            "mm3": float(v * float(np.prod(spacing))),
            "centroid_xyz_norm": [float(xx.mean() / W), float(yy.mean() / H), float(zz.mean() / D)],
            "bbox_voxels": [int(xx.min()), int(yy.min()), int(zz.min()),
                            int(xx.max()) + 1, int(yy.max()) + 1, int(zz.max()) + 1],
        })
    return rows


def run_eda(cfg: dict, data_root: str = "data/cads", synthetic: bool = False,
            out_dir: str = "results") -> dict:
    from .cads import load_native_mask, native_header_dhw
    from ..utils.common import load_cohort, split_cohort

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = cfg.get("dataset", {})
    parts = tuple(ds.get("seg_parts", [551]))
    subset_labels = ds.get("subset_labels") or {}

    samples = load_cohort(cfg, synthetic, data_root=data_root)
    tr, va, te = split_cohort(samples, cfg)

    # subset tag per sample for stratified reporting (volume_id has no subset info,
    # so attribute by label vector is unreliable — report pooled + per-split instead)
    shapes, spacings, comp_rows, per_vol, native_flags = [], [], [], [], []
    empty_vols = 0
    for s in samples:
        # native path preferred; fall back to resized sample (synthetic / missing files)
        try:
            if s.img_path and Path(s.img_path).exists() and Path(s.seg_path).exists():
                msk, spacing = load_native_mask(s.img_path, s.seg_path, parts, None)
                # NOTE: subset-specific labels need the subset name; VolumeSample
                # carries none, so EDA uses any-foreground in the requested parts
                # (size/location shape is what matters, not the exact filter).
                shape = msk.shape
                native_flags.append(True)
            else:
                raise FileNotFoundError("no source files")
        except Exception as e:
            print(f"[eda] native fallback to resized for {s.volume_id}: {e}")
            msk, spacing, shape = (s.mask > 0).astype(np.uint8), (1.0, 1.0, 1.0), s.volume.shape
            native_flags.append(False)
        try:
            hdr_shape, hdr_spacing = native_header_dhw(s.img_path) if s.img_path and Path(s.img_path).exists() else (shape, spacing)
        except Exception:
            hdr_shape, hdr_spacing = shape, spacing
        shapes.append(hdr_shape)
        spacings.append(hdr_spacing)
        comps = _components(msk, spacing)
        comp_rows.extend([{**c, "volume_id": s.volume_id} for c in comps])
        per_vol.append(len(comps))
        empty_vols += (len(comps) == 0)

    shapes = np.asarray(shapes)
    spacings = np.asarray(spacings)
    vols = np.asarray([c["voxels"] for c in comp_rows], dtype=float) if comp_rows else np.zeros(0)
    mm3 = np.asarray([c["mm3"] for c in comp_rows], dtype=float) if comp_rows else np.zeros(0)
    cents = np.asarray([c["centroid_xyz_norm"] for c in comp_rows]) if comp_rows else np.zeros((0, 3))

    summary = {
        "n_volumes": len(samples),
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "native_grids": bool(all(native_flags)) if native_flags else False,
        "shape_dhw": {k: v for k, v in zip(
            ["min", "median", "max"],
            [np.min(shapes, axis=0).tolist() if len(shapes) else None,
             np.median(shapes, axis=0).tolist() if len(shapes) else None,
             np.max(shapes, axis=0).tolist() if len(shapes) else None])},
        "spacing_dhw_mm": {k: v for k, v in zip(
            ["min", "median", "max"],
            [np.min(spacings, axis=0).tolist() if len(spacings) else None,
             np.median(spacings, axis=0).tolist() if len(spacings) else None,
             np.max(spacings, axis=0).tolist() if len(spacings) else None])},
        "n_objects": len(comp_rows),
        "volumes_with_no_objects": int(empty_vols),
        "objects_per_volume": {"mean": float(np.mean(per_vol)) if per_vol else 0.0,
                               "median": float(np.median(per_vol)) if per_vol else 0.0,
                               "max": int(np.max(per_vol)) if per_vol else 0},
        "object_voxels": {"median": float(np.median(vols)) if len(vols) else 0.0,
                          "p10": float(np.percentile(vols, 10)) if len(vols) else 0.0,
                          "p90": float(np.percentile(vols, 90)) if len(vols) else 0.0},
        "object_mm3": {"median": float(np.median(mm3)) if len(mm3) else 0.0,
                       "p10": float(np.percentile(mm3, 10)) if len(mm3) else 0.0,
                       "p90": float(np.percentile(mm3, 90)) if len(mm3) else 0.0},
        "centroid_xyz_norm_median": cents.mean(axis=0).tolist() if len(cents) else [0.5, 0.5, 0.5],
    }
    with open(out / "data_analysis_summary.json", "w") as f:
        import json
        json.dump(summary, f, indent=2)
    print("[eda] " + "; ".join([
        f"volumes={summary['n_volumes']} (tr/va/te={len(tr)}/{len(va)}/{len(te)})",
        f"native_DHW_shape_med={summary['shape_dhw']['median']}",
        f"spacing_med={summary['spacing_dhw_mm']['median']}mm",
        f"objects={summary['n_objects']} ({summary['objects_per_volume']['mean']:.1f}/vol, {empty_vols} empty)",
        f"size_med={summary['object_mm3']['median']:.0f}mm3 p10-p90={summary['object_mm3']['p10']:.0f}-{summary['object_mm3']['p90']:.0f}",
    ]))

    # --- figures ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].bar(["train", "val", "test"], [len(tr), len(va), len(te)])
    ax[0].set(title="Volumes per split", ylabel="volumes")
    if len(shapes):
        ax[1].boxplot([shapes[:, 0], shapes[:, 1], shapes[:, 2]])
        ax[1].set_xticks([1, 2, 3], ["D (slices)", "H", "W"])
        ax[1].set(title="Native grid shape spread (voxels)", ylabel="voxels")
    fig.tight_layout()
    fig.savefig(out / "eda_volumes_resolution.png", dpi=120)
    plt.close(fig)

    if len(vols):
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        ax[0].hist(np.log10(np.maximum(vols, 1)), bins=30)
        ax[0].set(title="Object size (log10 voxels)", xlabel="log10 voxels", ylabel="objects")
        ax[1].hist(per_vol, bins=np.arange(0, max(per_vol) + 2) - 0.5)
        ax[1].set(title="Objects per volume", xlabel="objects", ylabel="volumes")
        fig.tight_layout()
        fig.savefig(out / "eda_object_sizes.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
        for i, name in enumerate(["x (L–R)", "y (A–P)", "z (feet–head)"]):
            ax[i].hist(cents[:, i], bins=25, range=(0, 1))
            ax[i].set(title=f"Centroid {name}", xlabel="normalized", xlim=(0, 1))
        fig.suptitle("Where do target objects sit in the scan?")
        fig.tight_layout()
        fig.savefig(out / "eda_object_locations.png", dpi=120)
        plt.close(fig)
    print(f"[eda] figures -> {out}/eda_*.png")
    return summary
