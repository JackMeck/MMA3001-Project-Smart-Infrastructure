"""Smart Infrastructure: nowcasting room occupancy from environmental sensor data.

Estimates the number of people in one room from CO2, temperature, humidity, VOC and
PM2.5 readings of the Monash Smart Infrastructure sensors. A simple time-of-week
baseline is compared with linear regression and support-vector regression using
time-ordered cross-validation and a single held-out test period.

Modules:
    config: :class:`~smart_infrastructure.config.Config`, every setting in one place.
    data: loading, cleaning and aligning the sensor and occupancy data.
    models: the schedule baseline and the model factories.
    evaluation: metrics, chronological split, cross-validation, final test, timing.
    sensitivity: feature ablation, SVR settings, training-data requirement.
    floorspace: appendix study repeating the analysis on every floorspace.
    plots: test-result figures.
    pipeline: :func:`~smart_infrastructure.pipeline.run_pipeline`, the full workflow.

Example:
    >>> from smart_infrastructure import Config, run_pipeline
    >>> result = run_pipeline(Config(), "env.csv", "occ.csv")  # doctest: +SKIP
    >>> result["test"]  # doctest: +SKIP
"""

from . import config, data, evaluation, floorspace, models, pipeline, plots, sensitivity
from .config import Config
from .pipeline import run_pipeline

__version__ = "0.1.0"

__all__ = ["config", "data", "models", "evaluation", "sensitivity", "floorspace",
           "plots", "pipeline", "Config", "run_pipeline"]
