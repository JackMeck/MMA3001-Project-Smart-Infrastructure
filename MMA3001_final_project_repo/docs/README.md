# MMA3001 Project: Smart Infrastructure

[![tests](https://github.com/JackMeck/MMA3001-Project-Smart-Infrastructure/actions/workflows/tests.yml/badge.svg)](https://github.com/JackMeck/MMA3001-Project-Smart-Infrastructure/actions/workflows/tests.yml)

Nowcasting the number of people in a room from environmental sensor readings, using
data from the Monash Smart Infrastructure sensors. Individual project for MMA3001
(Numerical Methods and Machine Learning), Monash University.

## What this project does

**Engineering problem.** Buildings are heated, cooled and ventilated for the people
in them, but headcount sensors are expensive and are not installed everywhere. Cheap
environmental sensors (CO2, temperature, humidity, VOC, PM2.5) are common. This
project asks: *how well can the current number of people in one room be estimated
from the readings of one environmental sensor?*

**Approach.** Sensor readings are cleaned and placed on a regular 15-minute grid.
Three predictors are compared:

| Model | Uses sensors? | Purpose |
|---|---|---|
| Schedule baseline | No | Mean headcount for the same (weekday/weekend, hour) slot. A sensor model must beat this to show the sensors add information beyond the timetable. |
| Linear regression | Yes | Simple, interpretable, very cheap to run. |
| Support-vector regression (RBF) | Yes | Flexible non-linear model. |

Models are compared with time-ordered cross-validation and scored **once** on a
held-out final test period that is later in time than all training data. Further
studies cover feature ablation, SVR settings, training-data requirement,
computational cost (time and memory) and a repeat of the analysis on every room in
the dataset.

**Scope and limitations.**
- One sensor and one room. The pairing of sensor to room is an assumption, not
  confirmed by an official mapping; the all-floorspace appendix study shows how
  sensitive the result is to it.
- Results describe the study period (late February to mid June 2024) only.
- The task is a *nowcast* (current occupancy), not a forecast of future occupancy.
- Only a time-of-day / weekend schedule is used as a baseline; it does not know about
  public holidays or timetable changes.

## Inputs and outputs

**Inputs** (not included in this repository; the files are large):

| File | Format |
|---|---|
| `5EnvSensor_MayToDec2024_180kRows.csv` | Columns `sensorid`, `jsondata`. `jsondata` is a JSON list of variables, each with `(time, value)` pairs; `time` is Unix seconds. Variables used: Carbon dioxide (ppm), Temperature (°C), Humidity (%), Light Volatile Organic Compounds, Particulate matter 2.5. |
| `5occupancySensor_MayToDec2024_9MRows.csv` | An event stream with columns `floorspaceid`, `headcount` (people), `collecteddate` (ISO-8601 timestamp with UTC offset). |

**How bad input is handled**
- Sensor entries with an `errorCode`, or with no value, are discarded.
- Readings outside the study window are ignored.
- A CO2/temperature/humidity combination that makes up 5% or more of the records is a
  fixed placeholder, not a measurement, and is dropped.
- Short sensor gaps (up to 3 grid steps) are carried forward; longer gaps stay
  missing and those rows are not used.
- If there is no usable CO2 data, no occupancy events, no overlap between sensor and
  occupancy, or too few rows to split, a `ValueError` with a clear message is raised.

**Outputs**: a modelling table, cross-validation and final-test metrics (MAE and RMSE
in people, R², bias), ablation and sensitivity tables, timing and memory figures, a
floorspace scan, and a plot of the test results. `run_pipeline` returns all of these
as pandas objects, and writes them to a results folder as CSV/PNG/JSON when asked.

## Repository structure

```
.
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── pytest.ini
├── Final_Project.ipynb        analysis notebook
├── src/smart_infrastructure/
│   ├── config.py              all settings in one dataclass
│   ├── data.py                loading, cleaning, gridding, modelling table
│   ├── models.py              schedule baseline, linear and SVR pipelines
│   ├── evaluation.py          metrics, split, cross-validation, final test, timing
│   ├── sensitivity.py         ablation, SVR settings, training-data requirement
│   ├── floorspace.py          appendix: repeat on every room
│   ├── plots.py               test-result figures
│   ├── pipeline.py            run_pipeline: the full workflow
│   └── __main__.py            command-line entry point
├── docs/
│   ├── index.html             overview page
│   └── api/                   HTML API reference (generated with pdoc)
└── tests/                     pytest tests, test_report.html
```

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/JackMeck/MMA3001-Project-Smart-Infrastructure.git
cd MMA3001-Project-Smart-Infrastructure
pip install -r requirements.txt
pip install -e .
```

## Usage

Put the two CSV files in a `data/` folder (or point `--zip-dir` at the folder holding
the original `.zip` archives, and they are extracted for you).

From the command line:

```bash
python -m smart_infrastructure --data-dir data --out results
```

From Python:

```python
from smart_infrastructure import Config, run_pipeline

cfg = Config()   # defaults: sensor 6012002000869, room 941f0352-..., Feb-Jun 2024
result = run_pipeline(
    cfg,
    env_csv="data/5EnvSensor_MayToDec2024_180kRows.csv",
    occ_csv="data/5occupancySensor_MayToDec2024_9MRows.csv",
    out_dir="results",
)
print(result["test"])          # final held-out test metrics
print(result["improvement"])   # MAE improvement over the schedule baseline
```

Every setting (sensor, room, dates, grid, split, SVR settings) is a field of
`Config`; see the API reference for what each one does. The full analysis with its
discussion is in `Final_Project.ipynb`, and the generated tables are written to
`results/`.

## Testing

```bash
pytest --html=tests/test_report.html --self-contained-html --cov=smart_infrastructure
```

The tests use small synthetic files in the same layouts as the real data, so the real
data is not needed. They cover data cleaning, causal feature construction, the
chronological split and cross-validation folds (no use of future data), metrics against
hand-computed values, the models, the sensitivity studies, the floorspace appendix and
an end-to-end run. One test checks that the package reproduces the data-preparation
code of the original notebook exactly. The test report is `tests/test_report.html`, and
the tests also run automatically on every push through GitHub Actions
(`.github/workflows/tests.yml`).

## Documentation

Open `docs/index.html` for an overview, or `docs/api/smart_infrastructure.html` for the
full API reference. It is generated from the docstrings (Google style) with
[pdoc](https://pdoc.dev). To regenerate it after changing a docstring:

```bash
pdoc smart_infrastructure -o docs/api --docformat google
```

## Use of AI

Anthropic's Claude was used to help restructure the notebook code into this package,
write the tests and generate the documentation pages. The analysis, modelling choices
and conclusions are the author's; see the AI reflection in the written report for what
was used, how it was checked and where it was wrong or unhelpful.

## Licence

See [LICENSE](LICENSE).

## Data

Environmental and occupancy data from the Monash Smart Infrastructure sensors,
provided for the unit MMA3001. The data is not redistributed in this repository.
