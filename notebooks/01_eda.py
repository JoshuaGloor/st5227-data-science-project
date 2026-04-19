# -*- coding: utf-8 -*-
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
# # 01 Exploratory Data Analysis
#
# **Goal:** Understand the raw data well enough to make four design decisions:
# 1. Target definitions (which temporal aggregations of `bus_vol` should we choose).
# 2. Feature engineering for feature datasets `hdb`, `mrt`, `poi`, `bus_line`.
#
# For details about the CSVs of the LTSG dataset, it's also worth checking out the [project site](https://sites.google.com/view/ltsg/home) of the data source.
#
# Nothing is joined or modelled here. This exploration is the foundation for `02_features.py`.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import load_bus_vol, load_bus_line, load_hdb, load_mrt, load_poi
from src.eda_helpers import column_summary

# %% [markdown]
# ## 1 `bus_vol`: temporal schema and target design
# ### 1.1 Schema Exploration
#
# We begin by exploring the `bus_vol` dataset to understand its overall shape and contents.

# %%
bus_vol = load_bus_vol()
bus_vol.head(10)  # A sample

# %%
column_summary(bus_vol)

# %%
# So we know whether `hour` uses 0 or 24.
np.sort(bus_vol["hour"].drop_duplicates().sort_values())

# %% [markdown]
# We make the following observations which we cross check using the project site:
# - `day` is either weekday (*WD*) or weekend/holiday (*H*).
# - `hour` is 24h with "15 means the record numbers are gathered from 15:00 pm to 15:59 pm.". Range is $0, ..., 23$.
# - We see that we have 5018 unqiue `stop_id`s.
# - No missing values.
# - We have no `date` column, but only `month` (format 'yyyyMM') and whether it is a weekday or weekend/holiday.
#
# Next, we create some plots to understand the data distribution better.
#
# ### 1.2 Exploring the Data Distribution
#
# We start with a summary view. Below we plot hourly boardings (`in`) and alightings (`out`) per stop, split by weekday (`WD`) and holiday (`H`). Solid lines show the mean across stops at each hour; shaded bands show the interquartile range (Q1-Q3).

# %%
agg = (
    bus_vol.groupby(["day", "hour"])
    .agg(
        in_mean=("in", "mean"),
        in_q1=("in", lambda x: x.quantile(0.25)),
        in_q3=("in", lambda x: x.quantile(0.75)),
        out_mean=("out", "mean"),
        out_q1=("out", lambda x: x.quantile(0.25)),
        out_q3=("out", lambda x: x.quantile(0.75)),
    )
    .reset_index()
)

# Add wrap-around: for each day, duplicate hour 0 as hour 24
wrap = agg[agg["hour"] == 0].copy()
wrap["hour"] = 24
agg = pd.concat([agg, wrap], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True, constrained_layout=True)
COLORS = {"in": "tab:blue", "out": "tab:orange"}

for ax, day in zip(axes, ["WD", "H"]):
    sub = agg[agg["day"] == day]
    for metric in ["in", "out"]:
        c = COLORS[metric]
        # We plot steps because data point represents range. E.g., 15 is 15:00 - 15:59.
        ax.step(sub["hour"], sub[f"{metric}_mean"], where="post", color=c, linewidth=2, label=f"{metric} (mean)")
        # IQR steps.
        ax.fill_between(
            sub["hour"],
            sub[f"{metric}_q1"],
            sub[f"{metric}_q3"],
            step="post",
            color=c,
            alpha=0.15,
            label=f"{metric} (IQR)",
        )
    ax.set_title("Weekday" if day == "WD" else "Weekend/Holiday")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Passengers")
    ax.set_xticks(range(0, 25))
    ax.set_xticklabels([str(h) if h < 24 else "0" for h in range(25)])  # For wrap-around, we want hour 0, not 24.
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

plt.show()

# %% [markdown]
# We clearly see the difference in passenger numbers between days of the week and weekend/holiday. We can also observe rush hour peaks during week days and a more even landscape of the passenger numbers during the weekend/holiday.
#
# The small gap between the mean line and the upper edge of the IQR suggests right-skew of the ridership, both for `in` and `out`. This makes sense because we expect few high-volume stops (for example, stops closer to the CBD) to pull the mean up.
#
# Next, we want to confirm what the above plots suggests and observe the right-skewness of the ridership.
#
# For this we use a violin plot because we are curious about the shape of the whole distribution and the tail behavior in particular. Also, the violin plot will make it easier to spot differences between weekdays and weekends/holidays. Unlike a box plot, the violin shows where mass is concentrated (e.g. many quiet stops near zero vs a few very busy ones), making right-skew visible. Also, we can draw quartiles inside the violins to show the summary stats from a box plot; the only thing we lose is the explicit marking of individual outliers, which is less relevant here since the tail shape already shows how extreme they get.

# %%
# Create long format for hue, one row with `in` and `out` becomes two rows:
# one row with direction='in', passengers=pax_in and second row with
# direction='out', passenger=pax_out.
long = bus_vol.melt(id_vars=["day", "hour"], value_vars=["in", "out"], var_name="direction", value_name="passengers")

fig, axes = plt.subplots(1, 2, figsize=(13, 11), sharex=True, constrained_layout=True)

for ax, day in zip(axes, ["WD", "H"]):
    sub = long[long["day"] == day]
    sns.violinplot(
        data=sub,
        y="hour",
        x="passengers",
        hue="direction",
        ax=ax,
        orient="h",
        split=True,
        palette=COLORS,
        inner="quart",  # Show quartiles within the violins.
        cut=0,  # Avoid density past extremes.
        density_norm="width",  # 'width' (not 'count') bc we are interested in dist., ignoring passenger numbers.
    )
    ax.set_xscale("symlog")
    ax.set_title("Weekday" if day == "WD" else "Weekend/Holiday")
    ax.set_xlabel("Passengers")
    ax.set_ylabel("Hour")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="upper right")

plt.show()

# %% [markdown]
# This tells us a lot about our data. We notice that the ridership is very different between weekdays and weekends/holidays. In particular, we make the following observations:
#
# **Weekday (left)**
# - Hours 0 - 4: ridership is low to near zero; the network is basically idle.
# - From hour 5, we see that the numbers pick up. At hours 6 - 8 (i.e., 06:00 - 08:59) in particular, we notice long but slim right tails, suggesting a few very busy stops during rush hour.
# - Hours 9 - 15 (i.e., 09:00 - 15:59) show consistent traffic with little variation.
# - Traffic picks up again at hour 16 with another spike at hours 17 - 19 (i.e., 17:00 - 19:59) before traffic slows down towards the end of the day.
#
# **Weekend/Holiday (right)**
# - Hours 0 - 4: compared to weekdays, these early morning hours have noticeably fatter tails and more mass toward larger passengers counts. This can be explained by people coming home from night out, bars, etc.
# - From hour 5, we also see an increase of passengers, but the ridership is more evenly distributed compared to weekday traffic.
# - Throughout the day, the extreme tails are shorter and the mass of the distribution is more centered, suggesting that nearly all bus stops experience traffic and the ridership is more spread out.
#
# **Weekend/Holiday late night**
# Since it sticks out, we briefly mention hours 0 - 2 (i.e., 00:00 - 02:59) on weekends/holidays. We notice multiple modes and more mass away from zero than on weekdays, suggesting people stay out longer for leisure activities. It could be interesting to identify which stops have comparatively more activity during weekends/holidays than weekdays, e.g., stops near the airport or entertainment areas, but that is outside the scope of this project.
#
#
# From what we have seen in the previous two plots we also expect an asymmetry of passengers tapping `in` and tapping `out`. Since this is also an interesting question, we provide a brief plot below. It shows the total tap `out`s against the tap `in`s per bus stop.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True, constrained_layout=True)

for ax, day in zip(axes, ["WD", "H"]):
    stop_totals = bus_vol[bus_vol["day"] == day].groupby("stop_id")[["in", "out"]].sum().reset_index()

    ax.scatter(stop_totals["in"], stop_totals["out"], alpha=0.3, s=10)

    # Diagonal reference line (in == out)
    lim = max(stop_totals["in"].max(), stop_totals["out"].max())
    ax.plot([1, lim], [1, lim], "r--", linewidth=1, label="in = out")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total in")
    ax.set_ylabel("Total out")
    ax.set_title("Weekday" if day == "WD" else "Weekend/Holiday")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

plt.show()

# %% [markdown]
# We clearly see that there are asymmetric bus stops. Note that we plotted totals accross the whole day, it is natural to expect an even bigger asymmetry in the morning rush hour and evening rush hour. While this is interesting in its own right, modeling the directional flow is out of scope for this project, so we proceed with a combined `in` + `out` target capturing overall stop activity.

# %% [markdown]
# ### 1.3 Target construction sketch
#
# Since we are interested in predicting the busyness at bus stops, we define
# $$
#   \mathrm{volume} \coloneqq \mathrm{in} + \mathrm{out}.
# $$
#
# Based on what we observed above, we split our targets into different hourly periods with the goal of predicting the mean volume during each defined period. We define the following four scalar targets per stop.
# - `wd_am`: mean volume for weekday morning rush hours 6 - 8 (i.e., 06:00 - 08:59).
# - `wd_pm`: mean volume for weekday evening rush hours 16 - 19 (i.e., 16:00 - 19:59), where we extend the evening rush hour to include hour 19, consistent with later typically working hours in Singapore.
# - `wd_midday`: mean volume for weekday off-peak daytime hours 9 - 15 (i.e., 09:00 - 15:59).
# - `h_avg`: mean volume for weekend/holiday hours 6 - 22 (i.e., 06:00 - 22:59), leaving out early morning and late night hours because of their distinct patterns and low ridership.

# %% [markdown]
# ## 2. Feature sources: overview
#
# Beyond the bus volume data, we have four datasets that can contribute features: HDB (residential), MRT (rapid transit), POI (points of interest), and bus line (bus network connectivity). In each subsection below, we briefly review the columns and identify which features to carry forward into modeling.
#
# ### 2.1 `hdb`: candidate features

# %%
hdb = load_hdb()
column_summary(hdb, sample_values=3)

# %% [markdown]
# Based on this, we derive the following features per bus stop using `lat`/`lng` to find HBD blocks within 500m:
# - `hdb_units_500m`: sum of `total_dwelling_units` within 500m.<br>
#   *Reason*: total units as a proxy for people living nearby.
# - `hdb_rentals_500m`: sum of (`1room_rental` + `2room_rental` + `3room_rental` + `other_room_rental`) within 500m.<br>
#   *Reason*: rental units as proxy for lower-income households. Low-income citizens tend to be more dependent on public transport.
# - `hdb_commercial_500m`: count of blocks within 500m using `commercial == 'Y'`.<br>
#   *Reason*: commercial HDBs attract passenger activity.
#
# The 500m radius is somewhat arbitrary but it aligns with short walking distance guidelines for HDB amenities (see, e.g., [this HDB article](https://www.hdb.gov.sg/hdb-pulse/news/2026/residents-in-new-large-scale-bto-estates-to-enjoy-earlier-access-to-amenities), [this passage in particular](https://www.hdb.gov.sg/hdb-pulse/news/2026/residents-in-new-large-scale-bto-estates-to-enjoy-earlier-access-to-amenities#:~:text=Providing%20a%20more%20connected%20and%20comfortable%20living%20environment%20for%20residents)). We use 500m rather than a shorter radius because multiple bus stops are often close together, and passengers may choose between them depending on their destination.
#
# ### 2.2 `mrt`: candidate features

# %%
mrt = load_mrt()
column_summary(mrt, sample_values=3)

# %% [markdown]
# Based on this, we derive the following features per bus stop using `lat`/`lng` from the MRT stations (deduplicated to 184 unique stations, since interchange stations appear once per line):
#
# - `dist_nearest_mrt`: haversine distance in meters from each bus stop to its closest MRT station.<br>
#   *Reason*: proxy for bus stops used to reach the MRT, or to continue from the MRT to a final destination
# - `mrt_within_1km`: count of distinct MRT stations within 1km of each bus stop.<br>
#   *Reason*: proxy for rail network density; stops near multiple lines are in well-connected neighborhoods with higher overall transit demand.
#
# Unlike for the HDB data, we use a 1km radius (and continuous distance) for MRT because there are fewer MRTs overall and people are usually willing to walk further to a train station than to a bus stop.
#
# ### 2.3 `poi`: candidate features

# %%
poi = load_poi()
with pd.option_context("display.max_rows", None, "display.max_colwidth", None):  # To display all rows of the summary
    display(column_summary(poi).sort_index())  # Sorting index because there are many columns; easier to scan

# %%
# Alphabetical order of columns.
categories = {
    "education": ["primary_school", "school", "secondary_school", "university"],
    "food": ["bakery", "bar", "cafe", "food", "meal_takeaway", "restaurant"],
    "health": ["dentist", "doctor", "hospital", "pharmacy"],
    "leisure": ["amusement_park", "movie_theater", "museum", "park", "stadium", "tourist_attraction"],
    "shopping": ["clothing_store", "department_store", "grocery_or_supermarket", "shopping_mall", "supermarket"],
    "worship": ["church", "hindu_temple", "mosque", "place_of_worship"],
}

for cat, cols in categories.items():
    n = poi[cols].any(axis=1).sum()
    print(f"{cat}: {n} POIs ({n / len(poi) * 100:.1f}%)")

# %% [markdown]
# Based on this, we derive the following features per bus stop by aggregating POIs within 500m, grouped into five activity categories. Each POI is counted once per category, even if it has multiple matching flags.
#
# - `poi_education_500m`: count of education POIs within 500m; `"education": ["primary_school", "school", "secondary_school", "university"]`.<br>
#   *Reason*: schools drive predictable weekday early morning and afternoon ridership.
# - `poi_food_500m`: count of food/drink POIs within 500m; `"food": ["bakery", "bar", "cafe", "food", "meal_takeaway", "restaurant"]`.<br>
#   *Reason*: food destinations are strong drivers of (especially) midday and evening foot traffic.
# - `poi_health_500m`: count of health POIs within 500m; `"health": ["dentist", "doctor", "hospital", "pharmacy"]`.<br>
#   *Reason*: healthcare destinations attract non-commute trips, particularly from elderly residents who are often transit-dependent.
# - `poi_leisure_500m`: count of leisure/tourism POIs within 500m; `"leisure": ["amusement_park", "movie_theater", "museum", "park", "stadium", "tourist_attraction"]`.<br>
#   *Reason*: leisure destinations drive weekend/holiday ridership, matching the distinct patterns we observed in the violin plot.
# - `poi_worship_500m`: count of religious sites within 500m; `"worship": ["church", "hindu_temple", "mosque", "place_of_worship"]`.<br>
#     *Reason*: Singapore's diverse religious landscape drives predictable weekly traffic (e.g., Sunday services, Friday prayers), which may contribute to weekend/holiday ridership patterns.
# - `poi_shopping_500m`: count of shopping POIs within 500m; `"shopping": ["clothing_store", "department_store", "grocery_or_supermarket", "shopping_mall", "supermarket"]`.<br>
#   *Reason*: retail destinations attract daytime and weekend ridership.
#
# We skip many other columns (e.g., `rating`, `price_level`) to keep the feature set simple and in the interest of time. A more fine-grained treatment of POIs is left for future work.
#
# ### 2.4 `bus_line`: candidate features

# %%
bus_line = load_bus_line()
column_summary(bus_line, sample_values=3)

# %% [markdown]
# Based on this, we derive the following features per bus stop from the routing data:
#
# - `n_lines`: number of distinct bus lines serving the stop.<br>
#   *Reason*: stops served by more lines are better-connected hubs and tend to attract more passenger activity.
# - `is_terminal`: whether the stop is the first or last stop of any route (in either direction).<br>
#   *Reason*: terminals concentrate tapping in/out activity since every trip on that line starts or ends there.
#
# We skip `distance`, `sequence`, `operator`, and the first/last bus time columns. The timing columns could yield a "service span" feature (last bus minus first bus), but it is secondary to the line-count signal and omitted for simplicity.
#
# #### 2.4.1 A note on bus stop coordinates
#
# We expected to find `lat` and `lng` columns in `bus_line` (we probably should have checked this first), but they are not provided. Since the majority of our features are spatial (distance to MRT, POIs within 500m, etc.), coordinates are central to our modeling approach. Therefore, we fetch bus stop coordinates separately from the [LTA Bus Stop dataset on data.gov.sg](https://data.gov.sg/datasets/d_3f172c6feb3f4f92a2f47d93eed2908a/view). The actual join happens in `02_features.py`; approximately 23 of the ~5000 stops in `bus_vol` don't have a match in the current LTA data (likely retired or renumbered since the LTSG dataset was collected) and are dropped at that stage. Since this is a negligible fraction, we don't expect any meaningful impact.

# %% [markdown]
# ## 3 Summary of decisions
#
# Before moving to the feature enginneering notebook, `02_features`, we summarize:

# %% [markdown]
# ## Summary
#
# The tables below summarizes the targets and features defined above and will be computed in `02_features.py`.
#
# ### Targets (per stop)
#
# | Name | Definition |
# |---|---|
# | `wd_am` | Mean volume on weekdays, hours 6 - 8 (06:00 - 08:59) |
# | `wd_pm` | Mean volume on weekdays, hours 16 - 19 (16:00 - 19:59) |
# | `wd_midday` | Mean volume on weekdays, hours 9 - 15 (09:00 - 15:59) |
# | `h_avg` | Mean volume on weekends/holidays, hours 6 - 22 (06:00 - 22:59) |
#
# Volume is defined as `in + out`.
#
# ### Features (per stop)
#
# | Name | Source | Definition |
# |---|---|---|
# | `hdb_units_500m` | HDB | Sum of `total_dwelling_units` within 500m |
# | `hdb_rentals_500m` | HDB | Rental units within 500m |
# | `hdb_commercial_500m` | HDB | Count of blocks with `commercial == 'Y'` within 500m |
# | `dist_nearest_mrt` | MRT | Haversine distance (m) to closest MRT station |
# | `mrt_within_1km` | MRT | Count of distinct MRT stations within 1km |
# | `poi_education_500m` | POI | Count of education POIs within 500m |
# | `poi_food_500m` | POI | Count of food/drink POIs within 500m |
# | `poi_health_500m` | POI | Count of health POIs within 500m |
# | `poi_leisure_500m` | POI | Count of leisure/tourism POIs within 500m |
# | `poi_shopping_500m` | POI | Count of shopping POIs within 500m |
# | `poi_worship_500m` | POI | Count of religious-site POIs within 500m |
# | `n_lines` | bus_line | Number of distinct bus lines serving the stop |
# | `is_terminal` | bus_line | Whether the stop is a first or last stop in any route |

# %% [markdown]
#
