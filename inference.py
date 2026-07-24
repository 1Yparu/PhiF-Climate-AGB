from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


class InferenceError(RuntimeError):
    """Raised when a trained model cannot be loaded or used."""


def load_target_scaler(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load the optional shared target scaler used to restore original units.

    Expected JSON format::

        {
          "target_names": ["AGB"],
          "mean": [12.34],
          "scale": [5.67]
        }
    """
    scaler_path = Path(path)
    try:
        payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InferenceError(f"Target scaler not found: {scaler_path}") from exc
    except json.JSONDecodeError as exc:
        raise InferenceError(f"Invalid target scaler JSON: {scaler_path}") from exc

    names = payload.get("target_names")
    if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
        raise InferenceError("target_names must be a non-empty list of strings")

    mean = np.asarray(payload.get("mean"), dtype=float).reshape(-1)
    scale = np.asarray(payload.get("scale"), dtype=float).reshape(-1)
    if len(mean) != len(names) or len(scale) != len(names):
        raise InferenceError("Target scaler dimensions do not match target_names")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale == 0):
        raise InferenceError("Target scaler contains invalid values")
    return names, mean, scale


class ReleasedRegressor:
    """Uniform inference wrapper for the four published model types.

    Parameters
    ----------
    backend:
        One of ``xgboost``, ``lightgbm``, ``node``, or ``gandalf``.
    model_path:
        XGBoost model file, LightGBM model file, or the direct PyTorch Tabular
        saved-model directory for NODE/GANDALF.
    scaler_path:
        Optional path to a shared ``target_scaler.json``. It is required only
        when predictions must be returned in the original target units.
    feature_order:
        Optional explicit feature order. Usually unnecessary because
        PyTorch Tabular stores its data schema in ``datamodule.sav`` and tree
        model files generally preserve feature names when trained from a
        pandas DataFrame.
    """

    def __init__(
        self,
        backend: str,
        model_path: str | Path,
        *,
        scaler_path: str | Path | None = None,
        feature_order: Sequence[str] | None = None,
    ) -> None:
        self.backend = backend.strip().lower()
        if self.backend not in {"xgboost", "lightgbm", "node", "gandalf"}:
            raise InferenceError(f"Unsupported backend: {backend}")

        self.model_path = Path(model_path).expanduser().resolve()
        self.model = self._load_model()
        self.feature_order = list(feature_order) if feature_order is not None else self._infer_features()

        self.target_names: list[str] | None = None
        self.target_mean: np.ndarray | None = None
        self.target_scale: np.ndarray | None = None
        if scaler_path is not None:
            self.target_names, self.target_mean, self.target_scale = load_target_scaler(scaler_path)

    def _load_model(self) -> Any:
        if self.backend == "xgboost":
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise InferenceError("Install XGBoost with: pip install xgboost") from exc
            model = xgb.XGBRegressor()
            model.load_model(str(self.model_path))
            return model

        if self.backend == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError as exc:
                raise InferenceError("Install LightGBM with: pip install lightgbm") from exc
            return lgb.Booster(model_file=str(self.model_path))

        required = {
            "callbacks.sav",
            "config.yml",
            "custom_params.sav",
            "datamodule.sav",
            "model.ckpt",
        }
        if not self.model_path.is_dir():
            raise InferenceError(
                f"For {self.backend.upper()}, model_path must be the saved-model directory: {self.model_path}"
            )
        missing = sorted(name for name in required if not (self.model_path / name).exists())
        if missing:
            raise InferenceError(
                f"Incomplete PyTorch Tabular model directory; missing: {', '.join(missing)}"
            )
        try:
            from pytorch_tabular import TabularModel
        except ImportError as exc:
            raise InferenceError(
                "Install PyTorch Tabular with: pip install pytorch-tabular"
            ) from exc
        return TabularModel.load_model(str(self.model_path))

    def _infer_features(self) -> list[str] | None:
        if self.backend == "xgboost":
            try:
                names = self.model.get_booster().feature_names
                return list(names) if names else None
            except Exception:  # noqa: BLE001
                return None

        if self.backend == "lightgbm":
            try:
                names = self.model.feature_name()
                return list(names) if names else None
            except Exception:  # noqa: BLE001
                return None

        # PyTorch Tabular preserves its training schema in datamodule.sav.
        # Passing the input DataFrame directly is the most version-robust path.
        return None

    def _prepare_input(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise InferenceError("Input data must be a pandas DataFrame")

        X = data.copy()
        if self.feature_order:
            missing = [name for name in self.feature_order if name not in X.columns]
            if missing:
                raise InferenceError("Missing required features: " + ", ".join(missing))
            X = X.loc[:, self.feature_order]

        if X.empty:
            raise InferenceError("Input contains no feature columns")
        if X.isna().any().any():
            raise InferenceError("Input contains missing values")
        return X

    @staticmethod
    def _prediction_array(raw: Any, n_rows: int) -> tuple[np.ndarray, list[str] | None]:
        column_names: list[str] | None = None
        if isinstance(raw, pd.DataFrame):
            pred_cols = [c for c in raw.columns if str(c).endswith("_prediction")]
            if pred_cols:
                column_names = [str(c).removesuffix("_prediction") for c in pred_cols]
                raw = raw.loc[:, pred_cols]
            else:
                numeric = raw.select_dtypes(include=[np.number])
                if numeric.empty:
                    raise InferenceError("Model output has no numeric prediction columns")
                raw = numeric

        values = np.asarray(raw, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[0] != n_rows:
            raise InferenceError(f"Unexpected prediction shape: {values.shape}")
        return values, column_names

    def predict_standardized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict on the scale used to train the stored model."""
        X = self._prepare_input(data)
        raw = self.model.predict(X)
        values, inferred_names = self._prediction_array(raw, len(X))

        names = inferred_names
        if names is None and self.target_names is not None and len(self.target_names) == values.shape[1]:
            names = self.target_names
        if names is None:
            names = [f"target_{i + 1}" for i in range(values.shape[1])]

        return pd.DataFrame(
            values,
            columns=[f"{name}_prediction_standardized" for name in names],
            index=X.index,
        )

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict in original target units using ``target_scaler.json``."""
        if self.target_names is None or self.target_mean is None or self.target_scale is None:
            raise InferenceError(
                "Original-scale prediction requires scaler_path. "
                "Use predict_standardized() when no target scaler is supplied."
            )

        standardized = self.predict_standardized(data).to_numpy(dtype=float)
        if standardized.shape[1] != len(self.target_names):
            raise InferenceError("Prediction dimension does not match target scaler")
        original = standardized * self.target_scale + self.target_mean
        return pd.DataFrame(
            original,
            columns=[f"{name}_prediction" for name in self.target_names],
            index=data.index,
        )


def _parse_feature_order(text: str | None) -> list[str] | None:
    if text is None or not text.strip():
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one released regression model")
    parser.add_argument("--backend", required=True, choices=["xgboost", "lightgbm", "node", "gandalf"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--scaler", help="Optional shared target_scaler.json")
    parser.add_argument(
        "--features",
        help="Optional comma-separated feature order, mainly for legacy tree model files",
    )
    parser.add_argument(
        "--standardized-output",
        action="store_true",
        help="Return predictions without target inverse transformation",
    )
    args = parser.parse_args()

    X = pd.read_csv(args.input)
    model = ReleasedRegressor(
        args.backend,
        args.model_path,
        scaler_path=args.scaler,
        feature_order=_parse_feature_order(args.features),
    )
    prediction = model.predict_standardized(X) if args.standardized_output else model.predict(X)
    output = pd.concat([X.reset_index(drop=True), prediction.reset_index(drop=True)], axis=1)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Predictions written to: {output_path}")


if __name__ == "__main__":
    main()
