"""
scripts/download_dataset.py
===========================
Standalone script to download and parse the NASA PCoE CFRP dataset.

Usage:
    python scripts/download_dataset.py --output_dir ./data/parsed
"""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("downloader")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",    default="./data/raw",    help="Where to save/extract raw .mat files")
    parser.add_argument("--output_dir", default="./data/parsed", help="Where to write parquet output")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from src.data_loader import _download_with_progress, DATASET_URL, ZIP_FILENAME
    import zipfile, os

    raw_path = Path(args.raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    zip_path = raw_path / ZIP_FILENAME

    if not zip_path.exists() or args.force:
        logger.info("Downloading NASA PCoE CFRP dataset (~4.6 GB)…")
        _download_with_progress(DATASET_URL, zip_path)
        logger.info("Downloaded → %s", zip_path)
    else:
        logger.info("Zip already exists: %s", zip_path)

    extract_dir = raw_path / "composites_raw"
    if not extract_dir.exists() or args.force:
        logger.info("Extracting…")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        logger.info("Extracted → %s", extract_dir)

    mat_files = list(extract_dir.rglob("*.mat"))
    logger.info("Found %d .mat files — parsing to parquet…", len(mat_files))

    from src.nasa_parser import run_parse
    out_dir = Path(args.output_dir)
    run_parse(base_dir=extract_dir, output_dir=out_dir, raw_store_every=5)
    logger.info("Done! Parquets written to %s", out_dir)

if __name__ == "__main__":
    main()
