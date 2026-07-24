from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _clean_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. "
                "Pass overwrite=True to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _scaler_values(target_scaler: Any, n_targets: int) -> tuple[list[float], list[float]]:
    if not hasattr(target_scaler, "mean_") or not hasattr(target_scaler, "scale_"):
        raise TypeError(
            "target_scaler must be a fitted scaler exposing mean_ and scale_"
        )
    mean = np.asarray(target_scaler.mean_, dtype=float).reshape(-1)
    scale = np.asarray(target_scaler.scale_, dtype=float).reshape(-1)
    if len(mean) != n_targets or len(scale) != n_targets:
        raise ValueError(
            "The scaler target dimension does not match target_names"
        )
    return mean.tolist(), scale.tolist()


def _write_metadata(
    output_dir: Path,
    *,
    backend: str,
    model_path: str,
    feature_order: Sequence[str],
    target_names: Sequence[str],
    target_scaler: Any,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    feature_order = list(feature_order)
    target_names = list(target_names)
    if not feature_order or len(set(feature_order)) != len(feature_order):
        raise ValueError("feature_order must be non-empty and contain no duplicates")
    if not target_names or len(set(target_names)) != len(target_names):
        raise ValueError("target_names must be non-empty and contain no duplicates")

    mean, scale = _scaler_values(target_scaler, len(target_names))
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "backend": backend,
        "model_path": model_path,
        "feature_order": feature_order,
        "target_names": target_names,
        "target_scaler": {"mean": mean, "scale": scale},
    }
    if extra_metadata:
        metadata["model_information"] = extra_metadata

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def export_xgboost(
    model: Any,
    output_dir: str | Path,
    *,
    feature_order: Sequence[str],
    target_names: Sequence[str],
    target_scaler: Any,
    overwrite: bool = False,
) -> Path:
    """Export a fitted ``xgboost.XGBRegressor``."""
    output_dir = Path(output_dir)
    _clean_directory(output_dir, overwrite)
    model_file = output_dir / "model.json"
    model.save_model(str(model_file))
    params = model.get_params() if hasattr(model, "get_params") else {}
    _write_metadata(
        output_dir,
        backend="xgboost",
        model_path=model_file.name,
        feature_order=feature_order,
        target_names=target_names,
        target_scaler=target_scaler,
        extra_metadata={"parameters": params},
    )
    return output_dir


def export_lightgbm(
    model: Any,
    output_dir: str | Path,
    *,
    feature_order: Sequence[str],
    target_names: Sequence[str],
    target_scaler: Any,
    overwrite: bool = False,
) -> Path:
    """Export a fitted ``lightgbm.LGBMRegressor`` or ``lightgbm.Booster``."""
    output_dir = Path(output_dir)
    _clean_directory(output_dir, overwrite)
    model_file = output_dir / "model.txt"
    booster = model.booster_ if hasattr(model, "booster_") else model
    if not hasattr(booster, "save_model"):
        raise TypeError("Expected a fitted LightGBM regressor or Booster")
    booster.save_model(str(model_file))
    params = model.get_params() if hasattr(model, "get_params") else getattr(booster, "params", {})
    _write_metadata(
        output_dir,
        backend="lightgbm",
        model_path=model_file.name,
        feature_order=feature_order,
        target_names=target_names,
        target_scaler=target_scaler,
        extra_metadata={"parameters": params},
    )
    return output_dir


def export_pytorch_tabular(
    model: Any,
    output_dir: str | Path,
    *,
    backend: str,
    feature_order: Sequence[str],
    target_names: Sequence[str],
    target_scaler: Any,
    overwrite: bool = False,
) -> Path:
    """Export a fitted PyTorch Tabular NODE or GANDALF model.

    ``model.save_model`` stores the full PyTorch Tabular configuration and
    trained weights, so the model can later be restored with
    ``TabularModel.load_model`` without reconstructing the architecture.
    """
    backend = backend.lower()
    if backend not in {"node", "gandalf"}:
        raise ValueError("backend must be 'node' or 'gandalf'")

    output_dir = Path(output_dir)
    _clean_directory(output_dir, overwrite)
    saved_model_dir = output_dir / "saved_model"
    model.save_model(str(saved_model_dir))
    _write_metadata(
        output_dir,
        backend=backend,
        model_path=saved_model_dir.name,
        feature_order=feature_order,
        target_names=target_names,
        target_scaler=target_scaler,
    )
    return output_dir
