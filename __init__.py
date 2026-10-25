"""Room occupancy estimation from environmental sensors (MMA3001 project).

The package turns a single environmental sensor's readings (CO2, temperature,
humidity, VOC, PM2.5) into a predicted mean headcount for one floorspace in each
15-minute interval (a *nowcast*), and compares two sensor models (linear
regression and RBF support vector regression) against two no-sensor baselines
(a time-of-week schedule and a constant).

Modules
-------
config
    Constants: sensor/floorspace IDs, study window, split and CV settings.
data
    Loading, cleaning and gridding of sensor and occupancy files.
models
    Schedule baseline and scikit-learn pipeline builders.
evaluation
    Metrics, chronological train/test split and time-series cross-validation.
experiments
    Ablation, SVR tuning, data-requirement, timing and floorspace-sensitivity studies.
plotting
    Test-set figures.
pipeline
    End-to-end workflow that reproduces every table in the report.
"""

__version__ = "1.0.0"
