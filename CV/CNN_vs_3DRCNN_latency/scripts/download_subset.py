"""Download a SMALL CADS subset (never the full 22k volumes).

Needs: `hf auth login` + accepted gated-repo terms for mrmrx/CADS-dataset.

Example:
    python scripts/download_subset.py --subsets 0003_kits21 0004_lits --local-dir data/cads
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cads import download_subset  # noqa: E402
from src.utils.common import VERSION, log_version  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="mrmrx/CADS-dataset")
    ap.add_argument("--subsets", nargs="+", default=["0003_kits21", "0004_lits"])
    ap.add_argument("--local-dir", default="data/cads")
    ap.add_argument("--max-files", type=int, default=None,
                    help="cap total NIfTI files, spent in whole volumes "
                         "(1 image + one file per --parts). E.g. 120 ~ 60 vols. "
                         "Omit for full subsets.")
    ap.add_argument("--parts", type=int, nargs="+", default=[551],
                    help="Model-55X seg parts to fetch (default 551 = abdominal organs).")
    args = ap.parse_args()
    log_version("download")
    print(f"[dl] repo={args.repo} subsets={args.subsets} parts={args.parts} max_files={args.max_files}")
    print("[dl] NOTE: gated repo — run `hf auth login` and accept terms on the HF page first.")
    out = download_subset(args.repo, args.subsets, args.local_dir,
                          max_files=args.max_files, parts=tuple(args.parts))
    print(f"[dl] done -> {out}")


if __name__ == "__main__":
    main()
