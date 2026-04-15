# ST5227 — Applied Statistical Learning Project

Predicting Singapore bus stop passenger volume from spatial features engineered
from the [LTSG dataset](https://github.com/BlueSkyLT/siteselect_sg) (POIs, HDB
buildings, MRT stations).

## Project Overview

```
.
|-- README.md
|-- environment.yaml          # Conda environment file.
|-- requirements.txt
|-- setup.sh                  # Environment setup script.
|-- jupytext.toml
|-- data/                     # Raw CSVs (gitignored, see "Data" below).
|-- notebooks/                # Paired .py / .ipynb via jupytext.
|-- reports/                  # Generated figures and tables (gitignored).
`-- src/
    |-- __init__.py
    |-- data.py               # Load and join the 5 CSVs.
    |-- features.py           # Spatial feature engineering + target construction.
    |-- harness.py            # Experiment harness (CV across models / seeds / targets).
    `-- plots.py              # Plotting helpers.
```

## Data

Download `dataset.zip` from the
[LTSG repo](https://github.com/BlueSkyLT/siteselect_sg) and extract its contents
into `data/`. After extraction the directory should contain:

```
data/
|-- bus_line.csv
|-- bus_vol.csv
|-- hdb.csv
|-- mrt.csv
`-- poi.csv
```

## Setup

Make sure you have the following installed:

* miniconda

### Python Environment Setup and Jupyter Kernel

If you have a Bash-compatible shell, you can set up everything automatically:

```bash
source setup.sh
```

If you prefer or need to do the steps manually, the commands annotated in the
script under `[manual]` are reproduced here.

#### 1. Create the conda environment

```bash
conda env create -f environment.yaml
```

#### 2. Activate the environment

```bash
conda activate st5227
```

#### 3. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name=st5227 --display-name "Python (st5227)"
```

## Running the Code

After setup, start Jupyter:

```bash
jupyter notebook
```

Open notebooks under `notebooks/` and select the `Python (st5227)` kernel. As
per `jupytext.toml`, the `.py` files are the source of truth — open them via
right-click > Open With > Jupyter Notebook so they are paired with their
`.ipynb` counterparts. Only commit the `.py` files.

## Workflow Notes

- Format `.py` files before committing:
  ```bash
  black --line-length 119 <file.py>
  ```
- Format paired notebooks via jupytext:
  ```bash
  jupytext --pipe "black --line-length 119 -" <notebook.ipynb>
  ```
