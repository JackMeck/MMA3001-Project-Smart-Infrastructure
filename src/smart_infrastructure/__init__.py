"""Smart Infrastructure: nowcasting room occupancy from environmental sensor data.

The package estimates the number of people in a room (a *nowcast* of current
occupancy) from environmental sensor readings (CO2, temperature, humidity, VOC, PM2.5)
collected by the Monash Smart Infrastructure sensors.

Modules:
    config: :class:`~smart_infrastructure.config.Config` dataclass holding every
        dataset-specific setting.
    data: parsing, cleaning, resampling and pairing of environmental and occupancy data.
    features: causal (trailing-window) feature construction.
    models: baselines and candidate scikit-learn regression models.
    evaluation: splits, cross-validation, metrics, sensitivity studies and profiling.
    pipeline: end-to-end workflow (:func:`~smart_infrastructure.pipeline.run_monash`).

Example:
    >>> from smart_infrastructure import monash_config
    >>> from smart_infrastructure.pipeline import run_monash
    >>> cfg = monash_config(env_sensor_id="6012002000869",
    ...                     floorspace_id="941f0352-bb69-47d0-a9c5-ac29eb4f7319")
    >>> result = run_monash(cfg, out_dir="results/run1")  # doctest: +SKIP
"""

from . import config, data, evaluation, features, models, pipeline
from .config import Config
from .data import monash_config
from .features import FeatureSet, build_features
from .models import make_models, param_grids
from .pipeline import pairing_scan, run_from_clean, run_monash

__version__ = "0.1.0"

__all__ = [
    "config", "data", "features", "models", "evaluation", "pipeline",
    "Config", "monash_config", "FeatureSet", "build_features", "make_models",
    "param_grids", "run_from_clean", "run_monash", "pairing_scan",
]
