"""Data loading utilities for the LTSG dataset and bus stop coordinates.

The LTSG (Land & Transport Singapore) dataset is hosted at
https://github.com/BlueSkyLT/siteselect_sg and consists of five CSVs covering
points of interest, HDB buildings, MRT stations, bus stop passenger volumes,
and bus line routing.

Bus stop coordinates directly from data.gov.sg, API documentation at:
https://data.gov.sg/datasets/d_3f172c6feb3f4f92a2f47d93eed2908a/view

The first call to any of the ``load_*`` functions will download and extract the
zip (for LTSG) or do a GET (for bus stop coordinates) into the data directory
if it is not already present.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve
import json

import requests

import pandas as pd

import logging

logger = logging.getLogger(__name__)


def load_bus_stops(
    data_dir: Path | None = None,
    dataset_id: str = "d_3f172c6feb3f4f92a2f47d93eed2908a",
    local_geojson: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Load bus stop coordinates from LTA via data.gov.sg.

    Downloads the GeoJSON from data.gov.sg on first call and caches it as CSV
    in the data directory. Subsequent calls read from the cached CSV.

    If the API call fails (e.g., network issues, API changes), a local GeoJSON
    file can be provided via ``local_geojson`` as a fallback.

    Parameters
    ----------
    data_dir
        Directory to store the cached CSV. Defaults to ``<project_root>/data``.
    dataset_id
        The data.gov.sg dataset identifier for the LTA Bus Stop dataset.
    local_geojson
        Optional path to a local GeoJSON file (same schema as the data.gov.sg
        response). If provided, the GeoJSON is parsed from this file instead
        of being downloaded. Useful as a workaround if the API call fails.
    force
        If `True`, re-download (or re-parse the local file) even if the cached
        CSV exists.

    Returns
    -------
    pd.DataFrame
        One row per bus stop with columns:
        - stop_id: bus stop code (int, cast to match the LTSG datasets)
        - lat: latitude (float)
        - lng: longitude (float)
    """

    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "bus_stops.csv"

    if csv_path.exists() and not force:
        return pd.read_csv(csv_path)

    if local_geojson is not None:
        logger.info(f"Loading bus stops from local file {local_geojson} ...")
        with open(local_geojson) as f:
            geojson = json.load(f)
    else:
        logger.info(f"Downloading bus stops dataset {dataset_id} from data.gov.sg ...")

        poll_url = f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download"
        poll_response = requests.get(poll_url)
        poll_response.raise_for_status()
        poll_json = poll_response.json()

        if poll_json["code"] != 0:
            raise RuntimeError(f"data.gov.sg error: {poll_json['errMsg']}")

        geojson_url = poll_json["data"]["url"]
        geojson_response = requests.get(geojson_url)
        geojson_response.raise_for_status()
        geojson = geojson_response.json()

    rows = []
    skipped = 0
    for feature in geojson["features"]:
        raw_id = feature["properties"]["BUS_STOP_NUM"].strip()
        if not raw_id or not raw_id.isdigit() or len(raw_id) != 5:
            skipped += 1
            continue
        lng, lat = feature["geometry"]["coordinates"]
        stop_id = int(raw_id)
        rows.append({"stop_id": stop_id, "lat": lat, "lng": lng})

    if skipped:
        logger.info(f"Skipped {skipped} invalid bus stop entries")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    logger.info(f"Saved {len(df)} bus stops to {csv_path}")

    return df


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
        logger.info(f"Missing files: {', '.join(missing)}. Re-downloading...")

    if not zip_path.exists() or force:
        logger.info(f"Downloading {url} ...")
        urlretrieve(url, zip_path)

    logger.info(f"Extracting to {data_dir} ...")
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
