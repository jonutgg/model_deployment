"""Model training entrypoint for placement classification and salary regression."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from preprocessing_utils import prepare_training_data, save_bundle


@dataclass
class CandidateResult:
    model_name: str
    estimator: Pipeline
    metrics: dict[str, float]


def load_training_frame(feature_file: str, target_file: str) -> pd.DataFrame:
    features = pd.read_csv(feature_file)
    targets = pd.read_csv(target_file)
    merged = features.merge(targets, on="Student_ID", how="inner")
    if merged.empty:
        raise ValueError("Merged training dataframe is empty")
    return merged


def _fit_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    candidates: dict[str, object],
    scorer,
) -> CandidateResult:
    best: CandidateResult | None = None

    for model_name, base_estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("scale", RobustScaler()),
                ("model", clone(base_estimator)),
            ]
        )
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_valid)
        metrics = scorer(y_valid, prediction)

        candidate = CandidateResult(model_name=model_name, estimator=pipeline, metrics=metrics)
        if best is None or candidate.metrics["score"] > best.metrics["score"]:
            best = candidate

    if best is None:
        raise RuntimeError("No candidate models were trained")
    return best


def train_placement_model(X: pd.DataFrame, y: pd.Series, seed: int) -> CandidateResult:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1500, random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=260, random_state=seed),
    }

    def scorer(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        return {
            "score": weighted_f1,
            "f1_weighted": weighted_f1,
            "accuracy": accuracy_score(y_true, y_pred),
        }

    return _fit_candidates(X_train, y_train, X_valid, y_valid, candidates, scorer)


def train_salary_model(X: pd.DataFrame, y: pd.Series, seed: int) -> CandidateResult:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
    )

    candidates = {
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=320,
            random_state=seed,
            min_samples_leaf=2,
        ),
    }

    def scorer(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {
            "score": r2,
            "r2": r2,
            "mae": mae,
        }

    return _fit_candidates(X_train, y_train, X_valid, y_valid, candidates, scorer)


def save_training_outputs(
    output_dir: Path,
    features: list[str],
    placement_result: CandidateResult,
    salary_result: CandidateResult,
) -> None:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    placement_bundle = {
        "task": "placement",
        "trained_at_utc": timestamp,
        "model_name": placement_result.model_name,
        "features": features,
        "metrics": placement_result.metrics,
        "estimator": placement_result.estimator,
    }
    save_bundle(output_dir / "placement_model.joblib", placement_bundle)

    salary_bundle = {
        "task": "salary",
        "trained_at_utc": timestamp,
        "model_name": salary_result.model_name,
        "features": features,
        "metrics": salary_result.metrics,
        "estimator": salary_result.estimator,
    }
    save_bundle(output_dir / "salary_model.joblib", salary_bundle)

    report = {
        "trained_at_utc": timestamp,
        "placement": {
            "model": placement_result.model_name,
            "metrics": placement_result.metrics,
        },
        "salary": {
            "model": salary_result.model_name,
            "metrics": salary_result.metrics,
        },
        "num_features": len(features),
    }

    with (output_dir / "training_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train student outcome models")
    parser.add_argument("--features", default="A.csv", help="Path to feature CSV")
    parser.add_argument("--targets", default="A_targets.csv", help="Path to target CSV")
    parser.add_argument("--output", default="saved_models", help="Output model directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame = load_training_frame(args.features, args.targets)
    prepared = prepare_training_data(frame)

    placement_result = train_placement_model(prepared.X, prepared.y_placement, seed=args.seed)
    salary_result = train_salary_model(prepared.X, prepared.y_salary, seed=args.seed)

    save_training_outputs(
        output_dir=Path(args.output),
        features=list(prepared.X.columns),
        placement_result=placement_result,
        salary_result=salary_result,
    )

    print("Training complete")
    print(f"Placement model: {placement_result.model_name} | metrics={placement_result.metrics}")
    print(f"Salary model: {salary_result.model_name} | metrics={salary_result.metrics}")
    print(f"Artifacts written to: {args.output}")


if __name__ == "__main__":
    main()
