"""
Download and extract the MVTec AD dataset.

Purpose
-------
dataset profiling requires the MVTec AD dataset to be available under:

    data/raw/mvtec_ad/

This script supports two paths:

1. Automatic download from the configured archive URL.
2. Manual fallback if the URL fails.

Manual fallback
---------------
If the automatic download fails, download the archive manually from:

    https://www.mvtec.com/research-teaching/datasets/mvtec-ad

Then place the file here:

    data/raw/mvtec_anomaly_detection.tar.xz

Then rerun:

    python scripts/download_mvtec.py --extract-only

Why this script remains useful
------------------------------
Even if the dataset is manually downloaded, this script provides a reproducible,
documented extraction process.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defect_detection.utils.config import load_yaml_config  # noqa: E402
from defect_detection.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """
    Download a file with progress reporting.

    Parameters
    ----------
    url:
        URL to download.
    output_path:
        Destination file path.
    chunk_size:
        Download chunk size in bytes.

    Raises
    ------
    requests.HTTPError
        If the server returns an unsuccessful status code.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading MVTec AD archive...")
    logger.info("URL: %s", url)
    logger.info("Output: %s", output_path)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with output_path.open("wb") as file:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="Downloading",
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

    logger.info("Download complete.")


def extract_archive(archive_path: Path, raw_data_dir: Path, mvtec_root: Path) -> None:
    """
    Extract the MVTec AD tar.xz archive.

    Parameters
    ----------
    archive_path:
        Path to mvtec_anomaly_detection.tar.xz.
    raw_data_dir:
        Directory containing the archive.
    mvtec_root:
        Final expected dataset root directory.
    """
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Archive not found: {archive_path}\n"
            "Download manually from the official MVTec page and place it there."
        )

    logger.info("Extracting archive: %s", archive_path)
    logger.info("Extraction target: %s", raw_data_dir)

    with tarfile.open(archive_path, mode="r:xz") as tar:
        tar.extractall(path=raw_data_dir)

    extracted_root = raw_data_dir / "mvtec_anomaly_detection"

    if extracted_root.exists() and extracted_root != mvtec_root:
        if mvtec_root.exists():
            logger.info("Removing existing target directory before replacement: %s", mvtec_root)
            shutil.rmtree(mvtec_root)

        logger.info("Renaming %s to %s", extracted_root, mvtec_root)
        extracted_root.rename(mvtec_root)

    logger.info("Extraction complete.")


def dataset_already_exists(mvtec_root: Path) -> bool:
    """
    Check whether the dataset appears to already be extracted.

    Parameters
    ----------
    mvtec_root:
        Expected MVTec AD root directory.

    Returns
    -------
    bool
        True if expected category directories exist.
    """
    return (mvtec_root / "bottle").exists() and (mvtec_root / "screw").exists()


def main() -> int:
    """
    CLI entry point.

    Returns
    -------
    int
        Exit code. 0 means success.
    """
    parser = argparse.ArgumentParser(description="Download and extract MVTec AD.")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Skip download and only extract an existing local archive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download/extraction even if dataset appears to exist.",
    )

    args = parser.parse_args()

    config = load_yaml_config(PROJECT_ROOT / "configs/data/mvtec.yaml")

    raw_data_dir = PROJECT_ROOT / "data/raw"
    mvtec_root = PROJECT_ROOT / config["dataset"]["root_dir"]
    archive_path = PROJECT_ROOT / config["dataset"]["archive_path"]
    archive_url = config["dataset"]["archive_url"]

    if dataset_already_exists(mvtec_root) and not args.force:
        logger.info("MVTec AD appears to already exist at: %s", mvtec_root)
        logger.info("Use --force to re-download/re-extract.")
        return 0

    if not args.extract_only:
        try:
            download_file(archive_url, archive_path)
        except requests.RequestException as error:
            logger.error("Automatic download failed: %s", error)
            logger.error("Manual fallback:")
            logger.error("1. Open the official page: %s", config["dataset"]["official_page"])
            logger.error("2. Download mvtec_anomaly_detection.tar.xz")
            logger.error("3. Place it at: %s", archive_path)
            logger.error("4. Run: python scripts/download_mvtec.py --extract-only")
            return 1

    extract_archive(archive_path=archive_path, raw_data_dir=raw_data_dir, mvtec_root=mvtec_root)

    if not dataset_already_exists(mvtec_root):
        logger.error("Dataset extraction finished, but expected categories were not found.")
        logger.error("Expected at least: %s and %s", mvtec_root / "bottle", mvtec_root / "screw")
        return 1

    logger.info("MVTec AD dataset is ready at: %s", mvtec_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
