"""Feature engineering helpers for bus stop features and targets.

The helpers here take pandas DataFrames and return Series/DataFrames indexed
by ``stop_id``, ready to be joined onto a stop-level feature table.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000  # mean Earth radius in meters

# As discussed in 01_eda
POI_CATEGORIES: dict[str, list[str]] = {
    "education": ["primary_school", "school", "secondary_school", "university"],
    "food": ["bakery", "bar", "cafe", "food", "meal_takeaway", "restaurant"],
    "health": ["dentist", "doctor", "hospital", "pharmacy"],
    "leisure": [
        "amusement_park",
        "movie_theater",
        "museum",
        "park",
        "stadium",
        "tourist_attraction",
    ],
    "shopping": [
        "clothing_store",
        "department_store",
        "grocery_or_supermarket",
        "shopping_mall",
        "supermarket",
    ],
    "worship": ["church", "hindu_temple", "mosque", "place_of_worship"],
}


def _build_tree(coords: pd.DataFrame) -> BallTree:
    """Build a BallTree on haversine distance from a DataFrame of lat/lng (degrees)."""

    rad = np.deg2rad(coords[["lat", "lng"]].to_numpy())
    return BallTree(rad, metric="haversine")


def count_within_radius(
    stops: pd.DataFrame,
    entities: pd.DataFrame,
    radius_m: float,
    weights: pd.Series | None = None,
) -> pd.Series:
    """Count (or sum weights of) entities within ``radius_m`` of each stop.

    Parameters
    ----------
    stops
        DataFrame with ``stop_id``, ``lat``, ``lng`` columns.
    entities
        DataFrame with ``lat``, ``lng`` columns (the things to count).
    radius_m
        Search radius in meters.
    weights
        Optional Series aligned with ``entities``. If given, sums the weights
        of matched entities instead of counting them.

    Returns
    -------
    pd.Series
        Indexed by ``stop_id``, values are counts (or weighted sums).
    """

    tree = _build_tree(entities)
    stop_rad = np.deg2rad(stops[["lat", "lng"]].to_numpy())
    radius_rad = radius_m / EARTH_RADIUS_M

    indices = tree.query_radius(stop_rad, r=radius_rad)

    if weights is None:
        values = np.array([len(idx) for idx in indices])
    else:
        w = weights.to_numpy()
        values = np.array([w[idx].sum() for idx in indices])

    return pd.Series(values, index=stops["stop_id"].to_numpy(), name="count")


def distance_to_nearest(
    stops: pd.DataFrame,
    entities: pd.DataFrame,
) -> pd.Series:
    """Haversine distance in meters from each stop to its nearest entity.

    Parameters
    ----------
    stops
        DataFrame with ``stop_id``, ``lat``, ``lng`` columns.
    entities
        DataFrame with ``lat``, ``lng`` columns.

    Returns
    -------
    pd.Series
        Indexed by ``stop_id``, values are distances in meters.
    """

    tree = _build_tree(entities)
    stop_rad = np.deg2rad(stops[["lat", "lng"]].to_numpy())
    dist_rad, _ = tree.query(stop_rad, k=1)
    dist_m = dist_rad.flatten() * EARTH_RADIUS_M
    return pd.Series(dist_m, index=stops["stop_id"].to_numpy(), name="dist_m")


def sum_within_radius(
    stops: pd.DataFrame,
    entities: pd.DataFrame,
    radius_m: float,
    column: str,
) -> pd.Series:
    """Sum of ``entities[column]`` for entities within ``radius_m`` of each stop.

    A convenience wrapper around ``count_within_radius`` with weights.
    """

    return count_within_radius(stops, entities, radius_m, weights=entities[column])
