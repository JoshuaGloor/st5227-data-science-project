# ST5227 - Applied Statistical Learning - Project

Predicting Singapore bus stop passenger volume from spatial features engineered
from the [LTSG dataset](https://github.com/BlueSkyLT/siteselect_sg) from [Lan et al. (2022)](https://www.mdpi.com/2072-4292/14/15/3579).

The pdf version of the report can be found [here](report/report.pdf)

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

#### 3. Install project as editable package

```bash
pip install -e .
```

#### 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name=st5227 --display-name "Python (st5227)"
```

## Data

### LTSG dataset

The LTSG data is downloaded automatically by `src/data.py` on first use. Any of
the loader functions (`load_bus_vol`, `load_poi`, `load_hdb`, `load_mrt`,
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

#### Manual fallback

If the automatic download fails (e.g., network or other problems), download
`dataset.zip` manually from the [LTSG repo](https://github.com/BlueSkyLT/siteselect_sg) and extract it into
`data/`. The loader detects existing CSVs and skips the download.

### LTA bus stop coordinates

The LTSG dataset does not include coordinates for bus stops, so these are
fetched separately from the
[LTA Bus Stop dataset on data.gov.sg](https://data.gov.sg/datasets/d_3f172c6feb3f4f92a2f47d93eed2908a/view).
The loader `load_bus_stops` downloads the GeoJSON via the data.gov.sg API on
first use, parses it into a `stop_id`/`lat`/`lng` table, and caches it as
`data/bus_stops.csv`:

```python
from src.data import load_bus_stops
bus_stops = load_bus_stops()
```

Since the LTA dataset reflects the current bus network while the LTSG dataset
is a few years old, a small number of stops (~23 of ~5000) in `bus_vol` do not
have a match in the LTA data (likely retired or renumbered). These are dropped
at merge time.

#### Manual fallback

If the data.gov.sg API call fails, download the GeoJSON manually from the
[dataset page](https://data.gov.sg/datasets/d_3f172c6feb3f4f92a2f47d93eed2908a/view)
and pass the path to the loader:

```python
from pathlib import Path
from src.data import load_bus_stops

bus_stops = load_bus_stops(local_geojson=Path("/your/download/path/LTABusStop.geojson"))
```

## Usage

The notebooks in `notebooks/` are stored as `.py` files via
[jupytext](https://jupytext.readthedocs.io/). They can be opened as notebooks
in VS Code or your preferred editor. Select the `st5227` kernel when prompted.

For a quick run, start JupyterLab with:
```bash
jupyter lab
```
Right-click a `.py` file and select **Open With > Notebook** to run it as a notebook.

For VS Code, install the [jupytext extension](https://marketplace.visualstudio.com/items?itemName=congyiwu.vscode-jupytext) and right-click the `.py` file and select **Open as a Jupyter Notebook**.

[^1]: To preserve formatting when playing around, run `bash setup.sh --dev` to install pre-commit hooks.
