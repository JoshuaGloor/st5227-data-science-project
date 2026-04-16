# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (st5227)
#     language: python
#     name: st5227
# ---

# %% [markdown]
# # Loader Smoke Test

# %%
import sys

print(sys.executable)

# %%
import src

# %%
from src.data import load_bus_vol, load_bus_line, load_hdb, load_mrt, load_poi

# %%
for name, loader in [
    ("bus_vol", load_bus_vol),
    ("bus_line", load_bus_line),
    ("hdb", load_hdb),
    ("mrt", load_mrt),
    ("poi", load_poi),
]:
    df = loader()
    print(f"{name:>10s}: {df.shape[0]:>7,} rows × {df.shape[1]:>2} cols")

# %%
bus_vol = load_bus_vol()
bus_vol.head()

# %%
bus_vol.dtypes

# %%
bus_vol.describe()
