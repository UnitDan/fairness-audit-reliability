#!/usr/bin/env python3
"""Download third-party datasets into the local layout expected by the experiments."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ADULT_FILES = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    "adult.names": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
    "old.adult.names": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/old.adult.names",
    "Index": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/Index",
}
ML1M_ZIP = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"skip existing {destination.relative_to(ROOT)}")
        return
    print(f"download {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def download_adult() -> None:
    adult_dir = ROOT / "adult"
    for filename, url in ADULT_FILES.items():
        download(url, adult_dir / filename)


def download_ml1m() -> None:
    archive = ROOT / "ml-1m.zip"
    download(ML1M_ZIP, archive)
    target_dir = ROOT / "ml-1m"
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(ROOT)
    archive.unlink(missing_ok=True)
    required = ["users.dat", "ratings.dat", "movies.dat"]
    missing = [name for name in required if not (target_dir / name).exists()]
    if missing:
        raise RuntimeError(f"MovieLens archive did not contain expected files: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adult", action="store_true", help="download the UCI Adult dataset")
    parser.add_argument("--ml1m", action="store_true", help="download the MovieLens-1M dataset")
    args = parser.parse_args()

    if not args.adult and not args.ml1m:
        parser.error("select at least one dataset: --adult and/or --ml1m")
    if args.adult:
        download_adult()
    if args.ml1m:
        download_ml1m()


if __name__ == "__main__":
    main()
