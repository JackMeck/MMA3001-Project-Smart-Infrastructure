# MMA3001 Project - Room occupancy from environmental sensors

Predicts the mean headcount of one university floorspace in each 15-minute interval
from a single environmental sensor (CO2, temperature, humidity, VOC, PM2.5) - a
*nowcast* that could drive demand-controlled ventilation or fill gaps when a people
counter fails. Two sensor models (linear regression, RBF support vector regression)
are compared with two no-sensor baselines (time-of-week schedule, constant) using
time-ordered cross-validation and one final hold-out test.

> **Scope.** One floorspace, one sensor, one 3.5-month window (29 Feb - 14 Jun 2024).
> The dataset gives no sensor-to-room mapping, so results are an *empirical association*
> between a sensor and a floorspace; the five-floorspace study tests how much that matters.
> Output is an advisory estimate, not a safety-critical count.

Full write-up: the project report (submitted via Moodle).

## Inputs and outputs

| | Description |
| --- | --- |
| **Input** | Two CSV files (see [`data/README.md`](data/README.md)): environmental readings (`sensorid`, nested-JSON `jsondata`) and headcount events (`floorspaceid`, `headcount`, `collecteddate` UTC). |
| **Model inputs** | `co2` (ppm, ~400-650), `temperature` (deg C), `hour` (0-23.75, interval end), `weekend` (0/1). Others (`humidity`, `voc`, `pm2_5`, `co2_chg1h`) are tested and dropped. |
| **Output** | Predicted mean headcount (people) per 15-minute interval, clipped at 0. |
| **Invalid input** | Error-coded / value-less readings dropped; stuck "placeholder" readings (>= 5 % identical CO2/temp/humidity) removed; duplicate timestamps keep the last; gaps <= 45 min forward-filled, longer gaps dropped; remaining blanks median-imputed. Not validated for CO2 outside ~400-650 ppm, other sensors/rooms, or long sensor gaps. |

## Repository layout

```
.
├── src/occupancy/        # the package (all docstrings are Google style)
│   ├── config.py         #   constants: IDs, window, split, CV, SVR settings
│   ├── data.py           #   loading, cleaning, 15-min gridding, modelling table
│   ├── models.py         #   ScheduleBaseline + linear / SVR pipeline factories
│   ├── evaluation.py     #   metrics, chronological split, time-series CV
│   ├── experiments.py    #   ablation, SVR grid, data need, timing, floorspace study
│   ├── plotting.py       #   test-period figure
│   └── pipeline.py       #   end-to-end workflow (CLI entry point)
├── scripts/
│   ├── run_pipeline.py   # run everything
│   └── build_docs.sh     # regenerate HTML docs + pytest report
├── tests/                # pytest suite (synthetic data, runs in seconds) + test_report.txt
├── docs/                 # generated HTML API documentation (pdoc)
├── notebooks/run_in_colab.ipynb   # thin Colab launcher
├── data/                 # raw data goes here (not committed)
├── results/              # generated tables and figure (not committed)
├── LICENSE  requirements.txt  pyproject.toml
```

## Setup

Python >= 3.9.

```bash
git clone https://github.com/JackMeck/MMA3001-Project-Smart-Infrastructure.git
cd MMA3001-Project-Smart-Infrastructure
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                        # optional: makes `import occupancy` work anywhere
```

## Usage

1. Put the two CSV files in `data/` (or pass `--zip-dir` with the original zips).
2. Run the workflow:

```bash
python scripts/run_pipeline.py --data-dir data --output-dir results
# options: --zip-dir PATH   unpack original zips into --data-dir first
#          --skip-appendix  skip the five-floorspace study (avoids a second pass over the 9M-row file)
```

Tables are printed and written to `results/*.csv`; the test-period figure is
`results/test_results.png`. The models are deterministic, so figures reproduce exactly.

Use the package directly:

```python
from occupancy import config
from occupancy.data import load_environment, load_occupancy, add_co2_change, build_modelling_table
from occupancy.evaluation import chronological_split
from occupancy.models import final_models

env = add_co2_change(load_environment("data/5EnvSensor_MayToDec2024_180kRows.csv"))
occ = load_occupancy("data/5occupancySensor_MayToDec2024_9MRows.csv")
X_tr, y_tr, X_te, y_te = chronological_split(build_modelling_table(env, occ))
svr = final_models()["SVR"]().fit(X_tr, y_tr)
print(svr.predict(X_te)[:5])
```

## Mapping from the report to the code

| Report section | Code |
| --- | --- |
| 2. Inputs, outputs, data preparation | `occupancy.data` |
| 3-4. Methods and baselines | `occupancy.models` |
| 5. Validation (split, CV, metrics) | `occupancy.evaluation`, `tests/` |
| 6.1 Final comparison / test | `experiments.final_test`, `experiments.naive_benchmarks` |
| 6.2 Feature ablation | `experiments.feature_ablation`, `experiments.leave_one_out` |
| 6.3 SVR settings | `experiments.svr_grid` |
| 6.4 Training data needed | `experiments.data_requirement` |
| 6.5 Performance | `experiments.time_models` |
| 6.6 / Appendix B1 Floorspace mapping | `experiments.floorspace_scan`, `experiments.floorspace_sensitivity` |

## Tests and documentation

```bash
python -m pytest -v            # unit tests on synthetic data
scripts/build_docs.sh          # pdoc HTML into docs/ and tests/test_report.txt
```

The tests cover parsing and error handling, placeholder removal, grid alignment (each
reading goes to the *next* grid point), the no-future-information property of the CO2
change feature, held-state occupancy conversion, the baseline lookup, metric values,
the chronological split/gap, and expanding-window CV ordering.

## Method in brief

* **Validation:** all splits are chronological (autocorrelated data); 80 % train /
  20 % test with a 2-hour gap; five expanding-window CV folds on the training period
  guide every choice; the test set is scored once at the end.
* **Final models:** lean inputs (CO2, temperature, hour, weekend); SVR with
  C = 0.3, epsilon = 0.01 (cross-validated), default RBF gamma.
* **Metrics:** MAE (primary), RMSE, R2, bias.

## Limitations

One short test window, unconfirmed sensor location, a mostly empty room (MAE dominated
by near-zero intervals) and missed short peaks. See the report, Section 7.

## AI use

Claude (Anthropic) assisted with code and drafting; see the AI reflection in the report.

## Licence

MIT - see [`LICENSE`](LICENSE).
