"""Data loading utilities for the LTSG dataset.

The LTSG (Land & Transport Singapore) dataset is hosted at
https://github.com/BlueSkyLT/siteselect_sg and consists of five CSVs covering
points of interest, HDB buildings, MRT stations, bus stop passenger volumes,
and bus line routing.

The first call to any of the ``load_*`` functions will download and extract the
zip into the data directory if it is not already present.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


def download_data(
    data_dir: Path | None = None,
    url: str = "https://github.com/BlueSkyLT/siteselect_sg/raw/e631349/dataset.zip",
    force: bool = False,
) -> Path:
    """Download and extract the LTSG dataset zip.

    Parameters
    ----------
    data_dir
        Directory to extract the CSVs into. Defaults to ``<project_root>/data``.
    url
        (Commit pinned) URL of the zip.
    force
        If ``True``, re-download and re-extract even if the data already exists.

    Returns
    -------
    Path
        The directory containing the extracted CSVs.
    """

    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "dataset.zip"

    expected_files = [
        "bus_vol.csv",
        "bus_line.csv",
        "hdb.csv",
        "mrt.csv",
        "poi.csv",
    ]

    missing = [f for f in expected_files if not (data_dir / f).exists()]
    if not missing and not force:
        return data_dir
    if missing:
        print(f"Missing files: {', '.join(missing)}. Re-downloading...")

    if not zip_path.exists() or force:
        print(f"Downloading {url} ...")
        urlretrieve(url, zip_path)

    print(f"Extracting to {data_dir} ...")
    with zipfile.ZipFile(zip_path) as z:
        # Check if all files share a common top-level directory
        top_dirs = {name.split("/")[0] for name in z.namelist()}
        z.extractall(data_dir)

        # If there's a single wrapper folder, move contents up
        if len(top_dirs) == 1:
            wrapper = data_dir / top_dirs.pop()
            for item in wrapper.iterdir():
                item.rename(data_dir / item.name)
            wrapper.rmdir()

    return data_dir


def _load_csv(name: str, data_dir: Path | None, **download_kwargs) -> pd.DataFrame:
    resolved = download_data(data_dir=data_dir, **download_kwargs)
    return pd.read_csv(resolved / name)


def load_bus_vol(data_dir: Path | None = None, **download_kwargs) -> pd.DataFrame:
    return _load_csv("bus_vol.csv", data_dir, **download_kwargs)


def load_bus_line(data_dir: Path | None = None, **download_kwargs) -> pd.DataFrame:
    return _load_csv("bus_line.csv", data_dir, **download_kwargs)


def load_hdb(data_dir: Path | None = None, **download_kwargs) -> pd.DataFrame:
    return _load_csv("hdb.csv", data_dir, **download_kwargs)


def load_mrt(data_dir: Path | None = None, **download_kwargs) -> pd.DataFrame:
    return _load_csv("mrt.csv", data_dir, **download_kwargs)


def load_poi(data_dir: Path | None = None, **download_kwargs) -> pd.DataFrame:
    return _load_csv("poi.csv", data_dir, **download_kwargs)
