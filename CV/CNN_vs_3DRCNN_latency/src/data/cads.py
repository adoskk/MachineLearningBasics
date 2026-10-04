"""CADS NIfTI loading + task derivation (pure PyTorch project, no MONAI).

Dataset layout on HF (gated repo mrmrx/CADS-dataset):
    <subset>/images/<case>_0000.nii.gz
    <subset>/segmentations/<case>/<case>_part_55X.nii.gz   # 9 parts, one per specialist model
    <subset>/*.csv  (file manifest)

The `part_55X` files are outputs of 9 specialist models — see
https://github.com/murong-xu/CADS/blob/main/resources/info/labelmap.md.
For lesion-style tasks the money file is **part_551 (abdominal organs)**:
Spleen=1, Kidney R=2, Kidney L=3, Gallbladder=4, Liver=5, Stomach=6, ...
So KiTS21 -> label_ids [2,3] (kidneys, kidney-tumor proxy),
   LiTS    -> label_ids [5]    (liver, liver-tumor proxy).

What this module does:
  1. Download selected volumes (+ only their requested parts) via huggingface_hub.
  2. Pair images <-> per-case seg subdirs (tolerant to _0000 suffixes).
  3. HU-window + resize each volume to a fixed (D, H, W).
  4. Derive BOTH task labels from the SAME organ mask so the comparison is fair:
     - detection: 3D boxes [x1,y1,z1,x2,y2,z2] via connected components
     - classification: per-slice binary label (slice contains target organ?)

Honesty note: CADS masks are ORGAN pseudo-labels, not tumor boxes, and the
LiTS/KiTS test splits have no public tumor masks at all. The "lesion" task
here is an organ-region proxy: detect the kidney/liver, flag slices that
contain it. Same question for both models, fairly comparable.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None


@dataclass
class VolumeSample:
    volume_id: str
    volume: np.ndarray      # (D, H, W) float32, preprocessed
    mask: np.ndarray        # (D, H, W) uint8, resized with nearest
    boxes: np.ndarray       # (N, 6) float32 in voxel coords x1,y1,z1,x2,y2,z2
    slice_labels: np.ndarray  # (D,) int64, 1 if slice has >= thresh mask px
    img_path: str = ""      # source NIfTI (for native-resolution EDA); "" = synthetic
    seg_path: str = ""      # source seg case dir ("" = synthetic)


def hu_window(img: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    lo, hi = window
    img = np.clip(img, lo, hi).astype(np.float32)
    img = (img - lo) / max(hi - lo, 1e-6)  # [0, 1]
    return img


def resize_volume(vol: np.ndarray, shape: Tuple[int, int, int], is_mask: bool = False) -> np.ndarray:
    """Resize (D,H,W) with trilinear (image) or nearest (mask)."""
    t = torch.from_numpy(vol[None, None])  # (1,1,D,H,W)
    mode = "nearest" if is_mask else "trilinear"
    kw = {} if is_mask else {"align_corners": False}
    out = F.interpolate(t, size=shape, mode=mode, **kw)
    return out[0, 0].numpy()


def mask_to_boxes(
    mask: np.ndarray,
    min_voxels: int = 200,
    min_size: int = 5,
    max_boxes: int = 8,
) -> np.ndarray:
    """Connected components -> 3D boxes. Returns (N,6) x1,y1,z1,x2,y2,z2."""
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros((0, 6), dtype=np.float32)
    lab, n = ndi.label(binary)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = None, None
        zz, yy, xx = np.where(lab == i)
        if len(zz) < min_voxels:
            continue
        z1, y1, x1 = int(zz.min()), int(yy.min()), int(xx.min())
        z2, y2, x2 = int(zz.max()) + 1, int(yy.max()) + 1, int(xx.max()) + 1
        if min(z2 - z1, y2 - y1, x2 - x1) < min_size:
            continue
        boxes.append([x1, y1, z1, x2, y2, z2])
    if not boxes:
        return np.zeros((0, 6), dtype=np.float32)
    # largest-first, cap for memory
    boxes = sorted(boxes, key=lambda b: (b[3]-b[0])*(b[4]-b[1])*(b[5]-b[2]), reverse=True)[:max_boxes]
    return np.asarray(boxes, dtype=np.float32)


def mask_to_slice_labels(mask: np.ndarray, thresh: int = 10) -> np.ndarray:
    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    return (counts >= thresh).astype(np.int64)


def load_nifti_pair(
    img_path: str | Path,
    seg_case_dir: str | Path,
    target_shape: Tuple[int, int, int] = (64, 128, 128),
    hu_window_vals: Tuple[float, float] = (-160, 240),
    parts: Tuple[int, ...] = (551,),
    label_ids: Optional[List[int]] = None,
    slice_thresh: int = 10,
    min_voxels: int = 200,
    min_box: int = 5,
) -> VolumeSample:
    """Load image + requested seg parts, build ONE binary target mask.

    parts: which Model-55X outputs to use, e.g. (551,) = abdominal organs.
    label_ids: Model-55X label indices to keep, e.g. [2,3] kidneys, [5] liver.
               None = any foreground in the requested parts.
    """
    img = nib.load(str(img_path)).get_fdata(dtype=np.float32)
    # nibabel gives (H, W, D); transpose to (D, H, W) axial-first
    if img.ndim != 3:  # pragma: no cover - unexpected rank
        raise ValueError(f"expected 3D nifti, got {img.shape}")
    img = np.transpose(img, (2, 0, 1))
    img = hu_window(img, hu_window_vals)
    msk_bin = np.zeros(img.shape, dtype=np.uint8)
    part_files = _case_part_files(Path(seg_case_dir), parts)
    if not part_files:
        raise FileNotFoundError(f"no part files {parts} in {seg_case_dir}")
    ref_shape = None
    for pf in part_files:
        pm = nib.load(str(pf)).get_fdata().astype(np.int64)
        if pm.ndim != 3:
            raise ValueError(f"expected 3D nifti, got {pm.shape} in {pf}")
        pm = np.transpose(pm, (2, 0, 1))
        if ref_shape is None:
            ref_shape = pm.shape
        if pm.shape != ref_shape:
            raise ValueError(f"part geometry mismatch: {pf} {pm.shape} vs {ref_shape}")
        if label_ids is not None:
            hit = np.isin(pm, np.asarray(label_ids)).astype(np.uint8)
        else:
            hit = (pm > 0).astype(np.uint8)
        msk_bin = np.maximum(msk_bin, hit)
    if msk_bin.shape != img.shape:  # mask native grid differs from image (rare) -> resample via image grid
        raise ValueError(f"image/mask shape mismatch: {img.shape} vs {msk_bin.shape}")
    img_r = resize_volume(img, target_shape, is_mask=False)
    msk_r = (resize_volume(msk_bin.astype(np.float32), target_shape, is_mask=True) > 0.5).astype(np.uint8)
    boxes = mask_to_boxes(msk_r, min_voxels=min_voxels, min_size=min_box)
    # clip boxes to volume bounds (resize can push edges by 1px)
    D, H, W = target_shape
    if len(boxes):
        boxes[:, [0, 3]] = np.clip(boxes[:, [0, 3]], 0, W)
        boxes[:, [1, 4]] = np.clip(boxes[:, [1, 4]], 0, H)
        boxes[:, [2, 5]] = np.clip(boxes[:, [2, 5]], 0, D)
    sl = mask_to_slice_labels(msk_r, thresh=slice_thresh)
    vid = Path(img_path).name.replace(".nii.gz", "").replace(".nii", "")
    return VolumeSample(volume_id=vid, volume=img_r, mask=msk_r, boxes=boxes,
                        slice_labels=sl, img_path=str(img_path), seg_path=str(seg_case_dir))


def _load_dhw(path: str | Path, dtype=np.float32) -> np.ndarray:
    """Load a NIfTI and transpose to (D, H, W) axial-first, matching the loader."""
    arr = nib.load(str(path)).get_fdata(dtype=dtype)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D nifti, got {arr.shape} in {path}")
    return np.transpose(arr, (2, 0, 1))


def native_header_dhw(img_path: str | Path) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    """Native (D,H,W) grid shape + voxel spacing in mm, without loading voxels."""
    h = nib.load(str(img_path))
    sh = h.shape[:3]
    z = tuple(float(v) for v in h.header.get_zooms()[:3])
    return (sh[2], sh[0], sh[1]), (z[2], z[0], z[1])


def load_native_mask(img_path: str | Path, seg_case_dir: str | Path,
                     parts: Tuple[int, ...] = (551,),
                     label_ids: Optional[List[int]] = None) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Binary target mask at NATIVE resolution + (D,H,W) spacing. For EDA."""
    ref_shape, spacing = native_header_dhw(img_path)
    msk = np.zeros(ref_shape, dtype=np.uint8)
    for pf in _case_part_files(Path(seg_case_dir), parts):
        pm = _load_dhw(pf, np.int64)
        if pm.shape != ref_shape:
            raise ValueError(f"part geometry mismatch: {pf} {pm.shape} vs {ref_shape}")
        hit = np.isin(pm, np.asarray(label_ids)).astype(np.uint8) if label_ids is not None else (pm > 0).astype(np.uint8)
        msk = np.maximum(msk, hit)
    return msk, spacing


def _match_key(p: Path) -> str:
    """Filename key for pairing images <-> segmentations.

    Strips extensions AND nnU-Net-style modality suffixes, so
    `KITScase_00000_0000.nii.gz` (image) pairs with
    `KITScase_00000.nii.gz` (segmentation).
    """
    stem = re.sub(r"\.nii(\.gz)?$", "", p.name)
    stem = re.sub(r"_\d{4}$", "", stem)  # trailing _0000 modality id
    return stem


_PART_RE = re.compile(r"_part_(\d+)\.nii(\.gz)?$")


def _case_part_files(case_dir: Path, parts=None) -> List[Path]:
    """Sorted part files in a per-case seg subdir, filtered to requested parts."""
    files = sorted(case_dir.glob("*.nii*"))
    if parts is None:
        return files
    want = {str(p) for p in parts}
    out = []
    for f in files:
        m = _PART_RE.search(f.name)
        if m and m.group(1) in want:
            out.append(f)
    return out


def pair_files(local_dir: str | Path, subset: str) -> List[Tuple[Path, Path]]:
    """Pair images with their per-case seg subdir: segmentations/<case>/*_part_55X.nii.gz."""
    base = Path(local_dir) / subset
    img_dir, seg_dir = discover_local_layout(base)
    if img_dir is None or seg_dir is None:
        return []
    cases = {p.name: p for p in seg_dir.iterdir() if p.is_dir()}
    pairs = [(ip, cases[_match_key(ip)]) for ip in img_dir.glob("*.nii*")
             if _match_key(ip) in cases]
    return sorted(pairs, key=lambda t: t[0].name)


SEG_DIR_CANDIDATES = ("segmentations", "segmentationsTr", "labels", "labelsTr",
                        "masks", "annotations")


def discover_local_layout(subset_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find (images_dir, segs_dir) without hardcoding folder names.

    Source subsets came from different challenges (KiTS labelsTr, LiTS
    segmentations, …) and the HF reorganization is not uniform, so probe
    candidates instead of assuming `images/` + `segmentations/`.
    """
    if not subset_dir.exists():
        return None, None
    subdirs = [p for p in subset_dir.iterdir() if p.is_dir()]
    img = next((p for p in subdirs if p.name.lower() == "images"), None)
    if img is None:
        img = next((p for p in subdirs if "imag" in p.name.lower()), None)
    seg = None
    for name in SEG_DIR_CANDIDATES:
        p = subset_dir / name
        if p.is_dir() and p != img:
            seg = p
            break
    if seg is None:  # exactly one other nii-holding dir (at any depth) -> take it
        holders = [p for p in subdirs if p != img and next(p.rglob("*.nii*"), None)]
        if len(holders) == 1:
            seg = holders[0]
    return img, seg


def discover_repo_layout(repo_files: List[str], subset: str) -> Tuple[Optional[str], Optional[str]]:
    """Same discovery against HF repo-relative paths. Returns dir NAMES."""
    counts: dict = {}
    for f in repo_files:
        parts = f.split("/")
        if len(parts) >= 3 and parts[0] == subset and (f.endswith(".nii") or f.endswith(".nii.gz")):
            counts[parts[1]] = counts.get(parts[1], 0) + 1
    img = "images" if "images" in counts else next((d for d in counts if "imag" in d.lower()), None)
    seg = next((d for d in SEG_DIR_CANDIDATES if d in counts and d != img), None)
    if seg is None:
        rest = [d for d in counts if d != img]
        seg = rest[0] if len(rest) == 1 else None
    return img, seg


def describe_local(local_dir: str | Path, subsets: List[str]) -> str:
    """One-glance summary of what's actually on disk (for download debugging)."""
    base = Path(local_dir)
    lines = [f"local_dir={base.resolve()} (exists={base.exists()})"]
    if base.exists():
        top = sorted(p.name for p in base.iterdir())
        lines.append(f"top-level entries ({len(top)}): {top[:20]}")
    for s in subsets or []:
        b = base / s
        if not b.exists():
            lines.append(f"[{s}] MISSING directory")
            continue
        parts = []
        for p in sorted(b.iterdir()):
            if p.is_dir():
                n = sum(1 for _ in p.rglob("*.nii*"))
                extra = ""
                if p.name.lower() in ("segmentations", "labels", "labelstr", "masks"):
                    n_case = sum(1 for _ in p.iterdir() if _.is_dir())
                    extra = f", {n_case} cases" if n_case else ""
                parts.append(f"{p.name}/({n} nii{extra})")
            else:
                parts.append(p.name)
        lines.append(f"[{s}] " + ", ".join(parts[:12]))
        img_d, seg_d = discover_local_layout(b)
        lines.append(f"  -> detected images={img_d.name if img_d else None} "
                     f"segs={seg_d.name if seg_d else None}")
    return "\n".join(lines)


def _seg_key_and_part(seg_path: str) -> Tuple[str, Optional[str]]:
    """('KITScase_00000', '551') from '.../KITScase_00000/KITScase_00000_part_551.nii.gz'."""
    base = seg_path.split("/")[-1]
    m = _PART_RE.search(base)
    part = m.group(1) if m else None
    stem = _PART_RE.sub("", base)
    stem = re.sub(r"\.nii(\.gz)?$", "", stem)
    return _match_key(Path(stem)), part


def _repo_subset_pairs(repo_files: List[str], subset: str,
                        img_dir: str = "images", seg_dir: str = "segmentations",
                        parts: Tuple[int, ...] = (551,)
                        ) -> List[Tuple[str, List[str]]]:
    """Pair images with their per-case part files. Returns (img, [seg, ...]).

    Only pairs where ALL requested parts exist (else the mask would be partial).
    """
    want = [str(p) for p in parts]
    is_nii = lambda p: p.endswith(".nii") or p.endswith(".nii.gz")
    imgs = sorted(f for f in repo_files if f.startswith(f"{subset}/{img_dir}/") and is_nii(f))
    segmap: dict = {}
    for f in repo_files:
        if not f.startswith(f"{subset}/{seg_dir}/") or not is_nii(f):
            continue
        key, part = _seg_key_and_part(f)
        if part in want:
            segmap.setdefault(key, {})[part] = f
    pairs = []
    for ip in imgs:
        key = _match_key(Path(ip.split("/")[-1]))
        have = segmap.get(key, {})
        if all(p in have for p in want):
            pairs.append((ip, [have[p] for p in want]))
    return pairs


def select_capped_patterns(repo_files: List[str], subsets: List[str],
                            max_files: int,
                            layouts: Optional[dict] = None,
                            parts: Tuple[int, ...] = (551,)) -> Tuple[List[str], int, int]:
    """Even per-subset split of a total NIfTI file budget; pairs kept intact.

    Budget is counted in FILES but spent in WHOLE VOLUMES: each volume costs
    1 image + len(parts) segs. Tiny manifests (*.csv/*.md) ride free (KBs).
    Returns (allow_patterns, n_volumes, n_nifti_files).
    """
    layouts = layouts or {}
    per_vol = 1 + len(parts)
    per = max(max_files // max(len(subsets), 1) // per_vol, 0)
    patterns, n_vol, n_files = [], 0, 0
    for s in subsets:
        img_d, seg_d = layouts.get(s, ("images", "segmentations"))
        keep = _repo_subset_pairs(repo_files, s, img_d, seg_d, parts)[:per]
        for ip, sps in keep:
            patterns += [ip, *sps]
        n_vol += len(keep)
        n_files += len(keep) * per_vol
        patterns += [f"{s}/*.csv", f"{s}/*.md"]
    return patterns, n_vol, n_files


def download_subset(
    repo: str = "mrmrx/CADS-dataset",
    subsets: Optional[List[str]] = None,
    local_dir: str | Path = "data/cads",
    max_files: Optional[int] = None,
    parts: Tuple[int, ...] = (551,),
) -> Path:
    """Download selected volumes + only their requested seg parts.

    max_files: cap on total NIfTI files, spent in whole volumes
    (1 image + len(parts) segs each). E.g. parts=(551,) + max_files=120
    ~ 60 volumes — plenty for the default `n_volumes: 30` config.
    Omit for full subsets (note: uncapped KiTS+LiTS part_551 alone is
    ~1000 files; all 9 parts would be ~5000).
    """
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub not installed. pip install -r requirements.txt")
    subsets = subsets or ["0003_kits21"]
    # Discover the real folder names first: subsets came from different
    # challenges and don't uniformly use images/ + segmentations/.
    layouts: dict = {}
    try:
        from huggingface_hub import HfApi
        repo_files = list(HfApi().list_repo_files(repo_id=repo, repo_type="dataset"))
        for s in subsets:
            img_d, seg_d = discover_repo_layout(repo_files, s)
            layouts[s] = (img_d or "images", seg_d or "segmentations")
            print(f"[dl] {s}: images dir='{layouts[s][0]}' segs dir='{layouts[s][1]}'")
            if seg_d is None:
                print(f"[dl] WARNING: no segmentation folder found for {s} — "
                      "its volumes may have no public masks (e.g. challenge test splits).")
    except Exception as e:
        print(f"[dl] repo listing failed ({e}); falling back to images/ + segmentations/")
        repo_files = []
    if max_files is not None:
        if not repo_files:
            raise RuntimeError("max_files needs a repo listing, which failed — check HF auth.")
        patterns, n_vol, n = select_capped_patterns(repo_files, subsets, max_files, layouts, parts)
        print(f"[dl] file budget: {n_vol} volumes / {n} NIfTI files across {subsets} (cap {max_files})")
    else:
        patterns = []
        for s in subsets:
            img_d, seg_d = layouts.get(s, ("images", "segmentations"))
            patterns += [f"{s}/{img_d}/*"]
            # fnmatch '*' spans '/' so this reaches segmentations/<case>/*_part_55X
            patterns += [f"{s}/{seg_d}/*_part_{p}.nii.gz" for p in parts]
            patterns += [f"{s}/*.csv", f"{s}/*.md"]
    out = snapshot_download(
        repo_id=repo, repo_type="dataset",
        local_dir=str(local_dir), local_dir_use_symlinks=False,
        allow_patterns=patterns,
    )
    # verify: patterns that match nothing fail SILENTLY, so check the goods arrived
    n_img = sum(1 for s in subsets for _ in (Path(out) / s / "images").glob("*.nii*")) \
        if Path(out).exists() else 0
    print(f"[dl] verification: {n_img} NIfTI images under {out} for {subsets}")
    if n_img == 0:
        print("[dl] WARNING: 0 images downloaded — patterns matched nothing. Likely causes:\n"
              "  1. gated-repo access missing: run `hf auth login` AND accept terms on the HF dataset page\n"
              "  2. wrong subset names for this repo revision")
    return Path(out)


def load_volumes(
    local_dir: str | Path,
    subsets: List[str],
    n_volumes: int = 30,
    target_shape: Tuple[int, int, int] = (64, 128, 128),
    hu_window_vals: Tuple[float, float] = (-160, 240),
    slice_thresh: int = 10,
    min_voxels: int = 200,
    min_box: int = 5,
    seed: int = 42,
    parts: Tuple[int, ...] = (551,),
    subset_labels: Optional[dict] = None,
) -> List[VolumeSample]:
    """Load paired volumes, keeping only requested parts/labels per subset.

    subset_labels: {subset_name: [label ids]} using the requested parts'
    labelmap, e.g. {"0003_kits21": [2, 3]} for kidneys in Model-551.
    Missing entry (or None) = any foreground in the requested parts.
    """
    rng = np.random.default_rng(seed)
    all_pairs: List[Tuple[Path, Path, Optional[List[int]]]] = []
    for s in subsets:
        lab = (subset_labels or {}).get(s)
        all_pairs += [(ip, cd, lab) for ip, cd in pair_files(local_dir, s)]
    if not all_pairs:
        hint = ""
        for s in subsets:
            img_d, seg_d = discover_local_layout(Path(local_dir) / s)
            ni = sum(1 for _ in img_d.glob("*.nii*")) if img_d else 0
            ns = sum(1 for _ in seg_d.rglob("*.nii*")) if seg_d else 0
            if ni and seg_d is None:
                hint += (f"\n[{s}] has {ni} images but NO recognizable segmentation folder — "
                         "re-run download_subset.py (it discovers folder names), or the subset "
                         "may have no public masks (challenge test splits).")
            elif ni and ns:
                hint += (f"\n[{s}] has {ni} images + {ns} segs but 0 paired — "
                         "filename stems don't match; inspect with describe_local().")
        raise FileNotFoundError(
            f"no NIfTI pairs found under {local_dir} for {subsets}.\n"
            f"{describe_local(local_dir, subsets)}{hint}\n"
            "Fix: run scripts/download_subset.py from the PROJECT ROOT first "
            "(needs `hf auth login` + accepted gated-repo terms). If data was "
            "downloaded elsewhere, pass --data-root <dir> to the script."
        )
    idx = rng.permutation(len(all_pairs))[: min(n_volumes, len(all_pairs))]
    out, n_empty = [], 0
    for i in idx:
        ip, cd, lab = all_pairs[int(i)]
        try:
            s = load_nifti_pair(ip, cd, target_shape, hu_window_vals, parts, lab,
                                slice_thresh, min_voxels, min_box)
            if s.mask.sum() == 0:
                n_empty += 1
            out.append(s)
        except Exception as e:  # corrupt file -> skip, keep going
            print(f"[cads] skip {ip.name}: {e}")
    if n_empty:
        print(f"[cads] note: {n_empty}/{len(out)} volumes have EMPTY target masks "
              f"(parts={list(parts)} labels={subset_labels}) — boxes/slice-labels will be trivial.")
    return out
