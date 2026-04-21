"""FastAPI inference backend for student placement and salary predictions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from preprocessing_utils import REQUIRED_COLUMNS, load_bundle, prepare_inference_data


APP_TITLE = "Career Outcome Inference API"
MODEL_DIR = Path("saved_models")
PLACEMENT_BUNDLE = MODEL_DIR / "placement_model.joblib"
SALARY_BUNDLE = MODEL_DIR / "salary_model.joblib"


class StudentPayload(BaseModel):
    Student_ID: int = Field(..., ge=1)
    gender: str
    branch: str
    cgpa: float = Field(..., ge=0.0, le=10.0)
    tenth_percentage: float = Field(..., ge=0.0, le=100.0)
    twelfth_percentage: float = Field(..., ge=0.0, le=100.0)
    backlogs: int = Field(..., ge=0)
    study_hours_per_day: float = Field(..., ge=0.0, le=24.0)
    attendance_percentage: float = Field(..., ge=0.0, le=100.0)
    projects_completed: int = Field(..., ge=0)
    internships_completed: int = Field(..., ge=0)
    coding_skill_rating: int = Field(..., ge=1, le=5)
    communication_skill_rating: int = Field(..., ge=1, le=5)
    aptitude_skill_rating: int = Field(..., ge=1, le=5)
    hackathons_participated: int = Field(..., ge=0)
    certifications_count: int = Field(..., ge=0)
    sleep_hours: float = Field(..., ge=0.0, le=24.0)
    stress_level: int = Field(..., ge=1, le=10)
    part_time_job: str
    family_income_level: str
    city_tier: str
    internet_access: str
    extracurricular_involvement: str

    @model_validator(mode="after")
    def check_scores(self) -> "StudentPayload":
        if self.attendance_percentage < 30 and self.study_hours_per_day > 10:
            raise ValueError("attendance and study_hours combination looks invalid")
        return self


class InferenceResponse(BaseModel):
    student_id: int
    placement_label: str
    placement_score: float
    predicted_salary_lpa: float
    generated_at_utc: str


class BatchInferenceResponse(BaseModel):
    total_records: int
    accepted_records: int
    rejected_records: int
    outputs: list[dict[str, Any]]


class BundleStore:
    def __init__(self) -> None:
        self.placement: dict[str, Any] | None = None
        self.salary: dict[str, Any] | None = None

    def load(self) -> None:
        self.placement = load_bundle(PLACEMENT_BUNDLE) if PLACEMENT_BUNDLE.exists() else None
        self.salary = load_bundle(SALARY_BUNDLE) if SALARY_BUNDLE.exists() else None

    def status(self) -> dict[str, str]:
        return {
            "placement": "ready" if self.placement else "missing",
            "salary": "ready" if self.salary else "missing",
        }


store = BundleStore()
store.load()

app = FastAPI(title=APP_TITLE, version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_frame(payload: StudentPayload) -> pd.DataFrame:
    row = payload.model_dump()
    return pd.DataFrame([row], columns=REQUIRED_COLUMNS)


def _predict_placement(features: pd.DataFrame) -> tuple[str, float]:
    if not store.placement:
        raise HTTPException(status_code=503, detail="Placement model bundle unavailable")

    estimator = store.placement["estimator"]
    prediction = int(estimator.predict(features)[0])

    probability = 0.5
    if hasattr(estimator, "predict_proba"):
        probability = float(estimator.predict_proba(features)[0][1])

    label = "Placed" if prediction == 1 else "Not Placed"
    return label, probability


def _predict_salary(features: pd.DataFrame) -> float:
    if not store.salary:
        raise HTTPException(status_code=503, detail="Salary model bundle unavailable")

    estimator = store.salary["estimator"]
    predicted_salary = float(estimator.predict(features)[0])
    return round(max(predicted_salary, 0.0), 3)


def _full_inference(payload: StudentPayload) -> InferenceResponse:
    if not store.placement:
        raise HTTPException(status_code=503, detail="Placement model bundle unavailable")
    if not store.salary:
        raise HTTPException(status_code=503, detail="Salary model bundle unavailable")

    row_frame = _to_frame(payload)

    placement_features = prepare_inference_data(row_frame, store.placement["features"])
    salary_features = prepare_inference_data(row_frame, store.salary["features"])

    placement_label, placement_score = _predict_placement(placement_features)
    predicted_salary = _predict_salary(salary_features)

    return InferenceResponse(
        student_id=payload.Student_ID,
        placement_label=placement_label,
        placement_score=round(placement_score, 4),
        predicted_salary_lpa=predicted_salary,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/")
def index() -> dict[str, Any]:
    return {
        "service": APP_TITLE,
        "version": "2.0.0",
        "model_status": store.status(),
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "alive": True,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "model_status": store.status(),
    }


@app.post("/reload")
def reload_models() -> dict[str, Any]:
    store.load()
    return {"reloaded": True, "model_status": store.status()}


@app.post("/predict/single", response_model=InferenceResponse)
def predict_single(payload: StudentPayload) -> InferenceResponse:
    return _full_inference(payload)


@app.post("/predict/placement")
def predict_placement(payload: StudentPayload) -> dict[str, Any]:
    if not store.placement:
        raise HTTPException(status_code=503, detail="Placement model bundle unavailable")

    frame = _to_frame(payload)
    features = prepare_inference_data(frame, store.placement["features"])
    label, score = _predict_placement(features)

    return {
        "student_id": payload.Student_ID,
        "prediction": label,
        "confidence": round(score, 4),
        "prediction_type": "placement",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict/salary")
def predict_salary(payload: StudentPayload) -> dict[str, Any]:
    if not store.salary:
        raise HTTPException(status_code=503, detail="Salary model bundle unavailable")

    frame = _to_frame(payload)
    features = prepare_inference_data(frame, store.salary["features"])
    salary_lpa = _predict_salary(features)

    return {
        "student_id": payload.Student_ID,
        "prediction": salary_lpa,
        "confidence": 0.0,
        "prediction_type": "salary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict/batch", response_model=BatchInferenceResponse)
async def predict_batch(
    file: UploadFile = File(...),
    mode: Literal["placement", "salary", "both"] = "both",
) -> BatchInferenceResponse:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty")

    try:
        df = pd.read_csv(pd.io.common.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc

    outputs: list[dict[str, Any]] = []
    accepted = 0

    for _, row in df.iterrows():
        try:
            payload = StudentPayload(**row.to_dict())
            one_result: dict[str, Any] = {"student_id": payload.Student_ID}

            if mode in ("placement", "both"):
                placement = predict_placement(payload)
                one_result["placement_prediction"] = placement["prediction"]
                one_result["placement_confidence"] = placement["confidence"]

            if mode in ("salary", "both"):
                salary = predict_salary(payload)
                one_result["salary_prediction_lpa"] = salary["prediction"]

            outputs.append(one_result)
            accepted += 1
        except Exception as exc:
            outputs.append({"student_id": row.get("Student_ID", None), "error": str(exc)})

    total = len(df)
    return BatchInferenceResponse(
        total_records=total,
        accepted_records=accepted,
        rejected_records=total - accepted,
        outputs=outputs,
    )
