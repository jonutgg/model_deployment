"""Shared preprocessing and model bundle helpers for deployment workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Student_ID",
    "gender",
    "branch",
    "cgpa",
    "tenth_percentage",
    "twelfth_percentage",
    "backlogs",
    "study_hours_per_day",
    "attendance_percentage",
    "projects_completed",
    "internships_completed",
    "coding_skill_rating",
    "communication_skill_rating",
    "aptitude_skill_rating",
    "hackathons_participated",
    "certifications_count",
    "sleep_hours",
    "stress_level",
    "part_time_job",
    "family_income_level",
    "city_tier",
    "internet_access",
    "extracurricular_involvement",
]

CATEGORY_SPACE = {
    "gender": ["Male", "Female"],
    "part_time_job": ["No", "Yes"],
    "internet_access": ["No", "Yes"],
    "branch": ["CSE", "ECE", "IT", "CE"],
    "city_tier": ["Tier 1", "Tier 2", "Tier 3"],
    "family_income_level": ["Low", "Medium", "High"],
    "extracurricular_involvement": ["Low", "Medium", "High"],
}

BINARY_MAPS = {
    "gender": {"Male": 0, "Female": 1},
    "part_time_job": {"No": 0, "Yes": 1},
    "internet_access": {"No": 0, "Yes": 1},
}


@dataclass
class PreparedTrainingData:
    """Container for train-ready matrices and targets."""

    X: pd.DataFrame
    y_placement: pd.Series
    y_salary: pd.Series


def ensure_required_columns(df: pd.DataFrame) -> None:
    """Raise a readable error when required columns are missing."""
    missing = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _coerce_categories(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    for column, values in CATEGORY_SPACE.items():
        if column in clean.columns:
            clean[column] = (
                clean[column]
                .astype(str)
                .str.strip()
                .where(lambda s: s.isin(values), values[0])
            )
    return clean


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    enriched["core_skill_mean"] = enriched[
        ["coding_skill_rating", "communication_skill_rating", "aptitude_skill_rating"]
    ].mean(axis=1)

    enriched["academic_consistency"] = (
        enriched["tenth_percentage"]
        + enriched["twelfth_percentage"]
        + (enriched["cgpa"] * 10.0)
    ) / 3.0

    enriched["industry_readiness"] = (
        1.4 * enriched["projects_completed"]
        + 1.8 * enriched["internships_completed"]
        + 0.9 * enriched["certifications_count"]
        + 0.7 * enriched["hackathons_participated"]
    )

    enriched["discipline_signal"] = (
        enriched["attendance_percentage"] / 10.0
        + enriched["study_hours_per_day"]
        - enriched["backlogs"]
    )

    enriched["wellbeing_balance"] = (
        np.maximum(enriched["sleep_hours"], 1.0)
        / np.maximum(enriched["stress_level"], 1.0)
    )

    return enriched


def _encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()

    for column, mapping in BINARY_MAPS.items():
        encoded[column] = encoded[column].map(mapping).fillna(0).astype(int)

    multi_cols = [
        "branch",
        "city_tier",
        "family_income_level",
        "extracurricular_involvement",
    ]
    one_hot = pd.get_dummies(encoded[multi_cols], prefix=multi_cols, dtype=int)
    encoded = pd.concat([encoded.drop(columns=multi_cols), one_hot], axis=1)

    for column in CATEGORY_SPACE["branch"]:
        col_name = f"branch_{column}"
        if col_name not in encoded.columns:
            encoded[col_name] = 0

    for column in CATEGORY_SPACE["city_tier"]:
        col_name = f"city_tier_{column}"
        if col_name not in encoded.columns:
            encoded[col_name] = 0

    for column in CATEGORY_SPACE["family_income_level"]:
        col_name = f"family_income_level_{column}"
        if col_name not in encoded.columns:
            encoded[col_name] = 0

    for column in CATEGORY_SPACE["extracurricular_involvement"]:
        col_name = f"extracurricular_involvement_{column}"
        if col_name not in encoded.columns:
            encoded[col_name] = 0

    return encoded


def prepare_training_data(raw_df: pd.DataFrame) -> PreparedTrainingData:
    """Generate feature matrix and both targets from merged training data."""
    ensure_required_columns(raw_df)

    if "placement_status" not in raw_df.columns or "salary_lpa" not in raw_df.columns:
        raise ValueError("Training dataframe must include placement_status and salary_lpa")

    cleaned = _coerce_categories(raw_df)
    engineered = _engineer_features(cleaned)
    encoded = _encode_dataframe(engineered)

    y_placement = (raw_df["placement_status"].astype(str).str.lower() == "placed").astype(int)
    y_salary = pd.to_numeric(raw_df["salary_lpa"], errors="coerce").fillna(0.0)

    feature_df = encoded.drop(
        columns=["Student_ID", "placement_status", "salary_lpa"],
        errors="ignore",
    )

    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    feature_df = feature_df.reindex(sorted(feature_df.columns), axis=1)

    return PreparedTrainingData(X=feature_df, y_placement=y_placement, y_salary=y_salary)


def prepare_inference_data(raw_df: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    """Create model-ready features from user or API input dataframe."""
    ensure_required_columns(raw_df)

    cleaned = _coerce_categories(raw_df)
    engineered = _engineer_features(cleaned)
    encoded = _encode_dataframe(engineered)

    feature_df = encoded.drop(columns=["Student_ID"], errors="ignore")
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    for feature_name in expected_features:
        if feature_name not in feature_df.columns:
            feature_df[feature_name] = 0.0

    return feature_df[expected_features]


def save_bundle(path: str | Path, payload: dict[str, Any]) -> None:
    """Persist arbitrary model bundle using joblib."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, target)


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load persisted model bundle."""
    return joblib.load(path)
