# ST5227 - Applied Statistical Learning - Project

Predicting Singapore bus stop passenger volume from spatial features engineered
from the [LTSG dataset](https://github.com/BlueSkyLT/siteselect_sg) from [Lan et al. (2022)](https://www.mdpi.com/2072-4292/14/15/3579).

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

## Setup

### Prerequisites

* A [conda](https://docs.conda.io/en/latest/) installation, e.g., Miniconda (recommended) or Anaconda.

### Python Environment Setup and Jupyter Kernel

If you have a Bash-compatible shell, you can set up everything automatically[^1]:

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

## Data

The data is downloaded automatically by `src/data.py` on first use. Any of the
loader functions (`load_bus_vol`, `load_poi`, `load_hdb`, `load_mrt`,
`load_bus_line`) will fetch and extract the upstream zip into `data/` if it is
not already present. From a notebook:

```python
from src.data import load_bus_vol
bus_vol = load_bus_vol()
```

After the first call, `data/` will contain:

```
data/
|-- dataset.zip
|-- bus_line.csv
|-- bus_vol.csv
|-- hdb.csv
|-- mrt.csv
`-- poi.csv
```

### Manual fallback

If the automatic download fails (e.g., no network at grading time), download
`dataset.zip` manually from the
[LTSG repo](https://github.com/BlueSkyLT/siteselect_sg) and extract it into
`data/`. The loader detects existing CSVs and skips the download.


## Usage

The notebooks in `notebooks/` are stored as `.py` files via
[jupytext](https://jupytext.readthedocs.io/). They can be opened as notebooks
in VS Code or your preferred editor. Select the `st5227` kernel when prompted.

For a quick run, start JupyterLab with:
```bash
jupyter lab
```

[^1]: To preserve formatting when playing around, run `bash setup.sh --dev` to install pre-commit hooks.
