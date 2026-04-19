# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python (st5227)
#     language: python
#     name: st5227
# ---

# %% [markdown]
# # 02 Feature Engineering
#
# Construct per-stop features and targets from the raw datasets, following the definitions settled in `01_eda.py`. Outputs are saved to `data/features.csv` and `data/targets.csv` for use in `03_models.py`.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    load_bus_line,
    load_bus_stops,
    load_bus_vol,
    load_hdb,
    load_mrt,
    load_poi,
)
from src.features_helpers import (
    POI_CATEGORIES,
    count_within_radius,
    distance_to_nearest,
    sum_within_radius,
)

# %% [markdown]
# ## 1 Load data and join bus stop coordinates
#
# The `bus_vol` and `bus_line` datasets identify stops by `stop_id` only. We merge coordinates from the LTA bus stop dataset (fetched via `load_bus_stops`). The ~23 stops in `bus_vol` without a match in the current LTA data are dropped.

# %%
bus_vol = load_bus_vol()
bus_line = load_bus_line()
hdb = load_hdb()
mrt = load_mrt()
poi = load_poi()
bus_stops = load_bus_stops()

# %%
# Build the canonical stop table: one row per stop_id with coordinates.
stops = bus_vol[["stop_id"]].drop_duplicates().merge(bus_stops, on="stop_id", how="inner").reset_index(drop=True)
print(f"Stops with coordinates: {len(stops)} " f"(dropped {bus_vol['stop_id'].nunique() - len(stops)} without match)")

# %% [markdown]
# ## 2 Targets
#
# Four scalar targets per stop. Volume is `in + out` per hour-stop; we average over the hours defining each bucket.

# %%
bus_vol["volume"] = bus_vol["in"] + bus_vol["out"]

TARGET_BUCKETS = {
    "wd_am": {"day": "WD", "hours": range(6, 9)},  # 6, 7, 8
    "wd_pm": {"day": "WD", "hours": range(16, 20)},  # 16, 17, 18, 19
    "wd_midday": {"day": "WD", "hours": range(9, 16)},  # 9..15
    "h_avg": {"day": "H", "hours": range(6, 23)},  # 6..22
}


def compute_target(bus_vol: pd.DataFrame, day: str, hours: range) -> pd.Series:
    """Mean volume per stop for the given day type and hour range."""

    mask = (bus_vol["day"] == day) & (bus_vol["hour"].isin(hours))
    return bus_vol.loc[mask].groupby("stop_id")["volume"].mean()


# Realised that dataset does not indicate missing data with nulls, they just dropped that combination.
print("Information: some stops are missing data for particular day/hour combos")
print("We show details here but drop these missing combinations")
for name, spec in TARGET_BUCKETS.items():
    result = compute_target(bus_vol, **spec)
    print(f"{name}: {len(result)} stops with data")
print()

targets = pd.DataFrame({name: compute_target(bus_vol, **spec) for name, spec in TARGET_BUCKETS.items()})
targets = targets.dropna()
targets.index.name = "stop_id"
print(f"Stops with all 4 targets: {len(targets)}")

assert not targets.isna().any().any(), f"Unexpected NaNs in targets:\n{targets.isna().sum()[targets.isna().sum() > 0]}"

print(f"Targets shape: {targets.shape}")
targets.describe()

# %% [markdown]
# ## 3 HDB features
#
# Aggregate residential blocks within 500m of each stop.
# - `hdb_units_500m`: sum of `total_dwelling_units`
# - `hdb_rentals_500m`: rental units
# - `hdb_commercial_500m`: count of blocks with `commercial == 'Y'`

# %%
RENTAL_COLS = ["1room_rental", "2room_rental", "3room_rental", "other_room_rental"]
hdb = hdb.assign(
    rental_units=hdb[RENTAL_COLS].sum(axis=1),
    is_commercial=(hdb["commercial"] == "Y").astype(int),
)

hdb_units_500m = sum_within_radius(stops, hdb, 500, "total_dwelling_units")

hdb_rentals_500m = sum_within_radius(stops, hdb, 500, "rental_units")

hdb_commercial_500m = sum_within_radius(stops, hdb, 500, "is_commercial")

assert not hdb_rentals_500m.isna().any(), "Unexpected NaN in hdb_rentals"
assert not hdb_units_500m.isna().any(), "Unexpected NaN in hdb_units"
assert not hdb_commercial_500m.isna().any(), "Unexpected NaN in hdb_commercial"

hdb_features = pd.DataFrame(
    {
        "hdb_units_500m": hdb_units_500m,
        "hdb_rental_frac_500m": hdb_rentals_500m,
        "hdb_commercial_500m": hdb_commercial_500m,
    },
    index=stops["stop_id"],
)
hdb_features.describe()

# %% [markdown]
# ## 4 MRT features
#
# Deduplicate MRT stations (interchange stations appear once per line) before computing distances.

# %%
mrt_stations = mrt.drop_duplicates(subset="stop_id")[["stop_id", "lat", "lng"]]

dist_nearest_mrt = distance_to_nearest(stops, mrt_stations)
mrt_within_1km = count_within_radius(stops, mrt_stations, 1000)

assert not mrt_within_1km.isna().any(), "Unexpected NaN in mrt_within_1km"

mrt_features = pd.DataFrame(
    {"dist_nearest_mrt": dist_nearest_mrt, "mrt_within_1km": mrt_within_1km}, index=stops["stop_id"]
)
mrt_features.describe()

# %% [markdown]
# ## 5 POI features
#
# Group the fine-grained Google Places category flags into the five activity buckets defined in `01_eda.py`, then count POIs per bucket within 500m. Each POI is counted at most once per bucket, even if multiple flags match.

# %%
# Normalize all POI category flags to 0/1 once, regardless of whether stored
# as Y/N strings or bools.
all_flags: set[str] = set()
for flags in POI_CATEGORIES.values():
    all_flags.update(flags)

for flag in all_flags:
    if poi[flag].dtype == object:  # Y/N string
        poi[flag] = (poi[flag] == "Y").astype(int)
    else:  # already bool / int
        poi[flag] = poi[flag].astype(int)

poi = poi.assign(
    **{f"is_{category}": (poi[flags].sum(axis=1) > 0).astype(int) for category, flags in POI_CATEGORIES.items()}
)

poi_features = pd.DataFrame(
    {
        f"poi_{category}_500m": sum_within_radius(stops, poi, 500, f"is_{category}")
        for category, flags in POI_CATEGORIES.items()
    },
    index=stops["stop_id"],
)

poi_features.describe()

# %% [markdown]
# ## 6 Bus line features
#
# - `n_lines`: distinct bus lines serving each stop.
# - `is_terminal`: whether the stop is the first or last stop of any (line, direction) pair.

# %%
# Cast stop_id to match the int convention used elsewhere.
# Have to in a safe way because of unexpected values.
numeric = pd.to_numeric(bus_line["stop_id"], errors="coerce")
bad = bus_line.loc[numeric.isna(), "stop_id"]
print(f"Skipped {len(bad)} non-numeric stop_ids: {bad.unique().tolist()}")
bus_line = bus_line[numeric.notna()].copy()
bus_line["stop_id"] = numeric[numeric.notna()].astype(int)

n_lines = bus_line.groupby("stop_id")["line"].nunique()

# For each (line, direction), find first and last stop by sequence.
seq_bounds = bus_line.groupby(["line", "direction"])["sequence"].agg(["min", "max"])
bus_line_with_bounds = bus_line.merge(seq_bounds, on=["line", "direction"])
is_terminal_rows = (bus_line_with_bounds["sequence"] == bus_line_with_bounds["min"]) | (
    bus_line_with_bounds["sequence"] == bus_line_with_bounds["max"]
)
terminal_stop_ids = bus_line_with_bounds.loc[is_terminal_rows, "stop_id"].unique()

busline_features = pd.DataFrame(
    {
        "n_lines": n_lines,
        "is_terminal": n_lines.index.isin(terminal_stop_ids).astype(int),
    }
)
busline_features.index.name = "stop_id"
busline_features = busline_features.reindex(stops["stop_id"])
print(f"Stops with no bus_line data: {busline_features['n_lines'].isna().sum()} out of {len(busline_features)}")
busline_features = busline_features.fillna(0).astype(int)

busline_features.describe()

# %% [markdown]
# ## 7 Merge and save
#
# Combine all feature blocks on `stop_id`. We expect no missing values at this point; assertions below fail loudly if anything slipped through.

# %%
for name, df in [
    ("hdb_features", hdb_features),
    ("poi_features", poi_features),
    ("bus_features", busline_features),
    ("mrt_features", mrt_features),
]:
    idx = df.index if df.index.name == "stop_id" else df["stop_id"]
    dupes = idx.duplicated().sum()
    print(f"{name}: {dupes} duplicate stop_ids")

# %%
features = pd.concat([hdb_features, mrt_features, poi_features, busline_features], axis=1).reindex(stops["stop_id"])
features.index.name = "stop_id"

assert not features.isna().any().any(), (
    f"Unexpected NaNs in features:\n" f"{features.isna().sum()[features.isna().sum() > 0]}"
)

common_stops = targets.index.intersection(features.index)
print(f"stops: {len(stops)}  |  targets: {len(targets)}  |  features: {len(features)}  |  common: {len(common_stops)}")

features = features.loc[common_stops]
targets = targets.loc[common_stops]

print(f"Features shape: {features.shape}")
print(f"Feature columns: {list(features.columns)}")
features.describe()

# %%
out_dir = Path.cwd().parent / "data" if "__file__" not in globals() else Path(__file__).resolve().parents[1] / "data"

features.to_csv(out_dir / "features.csv")
targets.to_csv(out_dir / "targets.csv")
print(f"Saved features to {out_dir / 'features.csv'}")
print(f"Saved targets to {out_dir / 'targets.csv'}")
