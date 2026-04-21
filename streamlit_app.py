"""Local Streamlit app using on-disk model bundles (no API required)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from preprocessing_utils import REQUIRED_COLUMNS, load_bundle, prepare_inference_data


MODEL_DIR = Path("saved_models")
PLACEMENT_PATH = MODEL_DIR / "placement_model.joblib"
SALARY_PATH = MODEL_DIR / "salary_model.joblib"


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[dict | None, dict | None]:
    placement = load_bundle(PLACEMENT_PATH) if PLACEMENT_PATH.exists() else None
    salary = load_bundle(SALARY_PATH) if SALARY_PATH.exists() else None
    return placement, salary


def ui_header() -> None:
    st.title("Student Outcome Lab")
    st.caption("Run placement and salary inference directly from local artifacts")


def render_form() -> dict:
    col1, col2, col3 = st.columns(3)

    with col1:
        student_id = st.number_input("Student ID", min_value=1, value=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", ["CSE", "ECE", "IT", "CE"])
        city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

    with col2:
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.4, 0.1)
        tenth = st.slider("10th %", 0.0, 100.0, 78.0, 0.5)
        twelfth = st.slider("12th %", 0.0, 100.0, 77.0, 0.5)
        backlogs = st.number_input("Backlogs", min_value=0, value=0)
        attendance = st.slider("Attendance %", 0.0, 100.0, 82.0, 1.0)

    with col3:
        study_hours = st.slider("Study Hours", 0.0, 16.0, 4.5, 0.5)
        projects = st.number_input("Projects", min_value=0, value=3)
        internships = st.number_input("Internships", min_value=0, value=1)
        coding = st.slider("Coding Skill", 1, 5, 3)
        communication = st.slider("Communication Skill", 1, 5, 3)

    col4, col5 = st.columns(2)

    with col4:
        aptitude = st.slider("Aptitude Skill", 1, 5, 3)
        hackathons = st.number_input("Hackathons", min_value=0, value=1)
        certs = st.number_input("Certifications", min_value=0, value=2)
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
        stress = st.slider("Stress Level", 1, 10, 5)

    with col5:
        part_time_job = st.selectbox("Part-time Job", ["No", "Yes"])
        income = st.selectbox("Family Income", ["Low", "Medium", "High"])
        internet = st.selectbox("Internet Access", ["No", "Yes"])
        extracurricular = st.selectbox("Extracurricular", ["Low", "Medium", "High"])

    return {
        "Student_ID": int(student_id),
        "gender": gender,
        "branch": branch,
        "cgpa": float(cgpa),
        "tenth_percentage": float(tenth),
        "twelfth_percentage": float(twelfth),
        "backlogs": int(backlogs),
        "study_hours_per_day": float(study_hours),
        "attendance_percentage": float(attendance),
        "projects_completed": int(projects),
        "internships_completed": int(internships),
        "coding_skill_rating": int(coding),
        "communication_skill_rating": int(communication),
        "aptitude_skill_rating": int(aptitude),
        "hackathons_participated": int(hackathons),
        "certifications_count": int(certs),
        "sleep_hours": float(sleep_hours),
        "stress_level": int(stress),
        "part_time_job": part_time_job,
        "family_income_level": income,
        "city_tier": city_tier,
        "internet_access": internet,
        "extracurricular_involvement": extracurricular,
    }


def predict_local(row_dict: dict, placement_bundle: dict | None, salary_bundle: dict | None) -> dict:
    frame = pd.DataFrame([row_dict], columns=REQUIRED_COLUMNS)
    result = {"student_id": row_dict["Student_ID"]}

    if placement_bundle:
        features = prepare_inference_data(frame, placement_bundle["features"])
        estimator = placement_bundle["estimator"]
        label = int(estimator.predict(features)[0])
        confidence = 0.5
        if hasattr(estimator, "predict_proba"):
            confidence = float(estimator.predict_proba(features)[0][1])
        result["placement"] = "Placed" if label == 1 else "Not Placed"
        result["placement_confidence"] = confidence

    if salary_bundle:
        features = prepare_inference_data(frame, salary_bundle["features"])
        estimator = salary_bundle["estimator"]
        result["salary_lpa"] = float(estimator.predict(features)[0])

    return result


def main() -> None:
    st.set_page_config(page_title="Student Outcome Lab", layout="wide")
    ui_header()

    placement_bundle, salary_bundle = load_models()
    if not placement_bundle and not salary_bundle:
        st.error("No local models found. Run ml_pipeline.py first.")
        return

    tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

    with tab_single:
        payload = render_form()
        if st.button("Run Local Inference", use_container_width=True):
            output = predict_local(payload, placement_bundle, salary_bundle)
            c1, c2, c3 = st.columns(3)
            c1.metric("Student", str(output["student_id"]))
            c2.metric("Placement", output.get("placement", "N/A"))
            c3.metric("Salary (LPA)", f"{output.get('salary_lpa', 0.0):.2f}")

            if "placement_confidence" in output:
                st.progress(min(max(output["placement_confidence"], 0.0), 1.0))
                st.caption(f"Placement confidence: {output['placement_confidence']:.2%}")

    with tab_batch:
        st.write("Upload a CSV with all required student columns.")
        uploaded = st.file_uploader("Batch CSV", type=["csv"])
        if uploaded is not None:
            batch = pd.read_csv(uploaded)
            st.dataframe(batch.head(), use_container_width=True)

            if st.button("Process Batch", use_container_width=True):
                outputs = []
                for _, row in batch.iterrows():
                    try:
                        row_dict = {k: row[k] for k in REQUIRED_COLUMNS}
                        outputs.append(predict_local(row_dict, placement_bundle, salary_bundle))
                    except Exception as exc:
                        outputs.append({"student_id": row.get("Student_ID", None), "error": str(exc)})

                result_df = pd.DataFrame(outputs)
                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    "Download Results",
                    result_df.to_csv(index=False),
                    file_name=f"local_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
