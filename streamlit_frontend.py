"""Streamlit client for interacting with the FastAPI inference backend."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = "http://localhost:8000"


def collect_form_input() -> dict:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        student_id = st.number_input("Student ID", min_value=1, value=101)
        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", ["CSE", "ECE", "IT", "CE"])
        city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

    with c2:
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.2, 0.1)
        tenth_percentage = st.slider("10th %", 0.0, 100.0, 76.0, 0.5)
        twelfth_percentage = st.slider("12th %", 0.0, 100.0, 74.0, 0.5)
        backlogs = st.number_input("Backlogs", min_value=0, value=0)

    with c3:
        study_hours = st.slider("Study Hours", 0.0, 16.0, 4.0, 0.5)
        attendance = st.slider("Attendance %", 0.0, 100.0, 80.0, 1.0)
        projects = st.number_input("Projects", min_value=0, value=2)
        internships = st.number_input("Internships", min_value=0, value=1)

    with c4:
        coding = st.slider("Coding", 1, 5, 3)
        communication = st.slider("Communication", 1, 5, 3)
        aptitude = st.slider("Aptitude", 1, 5, 3)
        hackathons = st.number_input("Hackathons", min_value=0, value=1)

    d1, d2 = st.columns(2)
    with d1:
        certifications = st.number_input("Certifications", min_value=0, value=1)
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
        stress = st.slider("Stress", 1, 10, 5)
        part_time_job = st.selectbox("Part-time Job", ["No", "Yes"])

    with d2:
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
        internet = st.selectbox("Internet", ["No", "Yes"])
        extracurricular = st.selectbox("Extracurricular", ["Low", "Medium", "High"])

    return {
        "Student_ID": int(student_id),
        "gender": gender,
        "branch": branch,
        "cgpa": float(cgpa),
        "tenth_percentage": float(tenth_percentage),
        "twelfth_percentage": float(twelfth_percentage),
        "backlogs": int(backlogs),
        "study_hours_per_day": float(study_hours),
        "attendance_percentage": float(attendance),
        "projects_completed": int(projects),
        "internships_completed": int(internships),
        "coding_skill_rating": int(coding),
        "communication_skill_rating": int(communication),
        "aptitude_skill_rating": int(aptitude),
        "hackathons_participated": int(hackathons),
        "certifications_count": int(certifications),
        "sleep_hours": float(sleep_hours),
        "stress_level": int(stress),
        "part_time_job": part_time_job,
        "family_income_level": family_income,
        "city_tier": city_tier,
        "internet_access": internet,
        "extracurricular_involvement": extracurricular,
    }


def call_api(base_url: str, path: str, method: str = "GET", **kwargs):
    url = f"{base_url.rstrip('/')}{path}"
    if method == "GET":
        return requests.get(url, timeout=15, **kwargs)
    return requests.post(url, timeout=60, **kwargs)


def single_prediction_ui(base_url: str) -> None:
    st.subheader("Single Student Prediction")
    payload = collect_form_input()

    c1, c2, c3 = st.columns(3)
    with c1:
        run_combined = st.button("Predict Both", use_container_width=True)
    with c2:
        run_placement = st.button("Placement Only", use_container_width=True)
    with c3:
        run_salary = st.button("Salary Only", use_container_width=True)

    if run_combined:
        response = call_api(base_url, "/predict/single", method="POST", json=payload)
        if response.ok:
            data = response.json()
            x1, x2, x3 = st.columns(3)
            x1.metric("Placement", data["placement_label"])
            x2.metric("Placement Score", f"{data['placement_score']:.2%}")
            x3.metric("Salary LPA", f"{data['predicted_salary_lpa']:.2f}")
            st.caption(f"Generated: {data['generated_at_utc']}")
        else:
            st.error(response.text)

    if run_placement:
        response = call_api(base_url, "/predict/placement", method="POST", json=payload)
        if response.ok:
            data = response.json()
            st.metric("Placement", data["prediction"])
            st.progress(min(max(data["confidence"], 0.0), 1.0))
            st.caption(f"Confidence: {data['confidence']:.2%}")
        else:
            st.error(response.text)

    if run_salary:
        response = call_api(base_url, "/predict/salary", method="POST", json=payload)
        if response.ok:
            data = response.json()
            st.metric("Predicted Salary (LPA)", f"{data['prediction']:.2f}")
        else:
            st.error(response.text)


def batch_prediction_ui(base_url: str) -> None:
    st.subheader("Batch Prediction via API")
    mode = st.selectbox("Mode", ["both", "placement", "salary"])
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        return

    preview = pd.read_csv(uploaded)
    st.dataframe(preview.head(), use_container_width=True)

    if st.button("Run Batch Request", use_container_width=True):
        uploaded.seek(0)
        files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
        response = call_api(base_url, "/predict/batch", method="POST", files=files, params={"mode": mode})

        if not response.ok:
            st.error(response.text)
            return

        data = response.json()
        a, b, c = st.columns(3)
        a.metric("Total", str(data["total_records"]))
        b.metric("Accepted", str(data["accepted_records"]))
        c.metric("Rejected", str(data["rejected_records"]))

        result_df = pd.DataFrame(data["outputs"])
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            "Download Batch Output",
            result_df.to_csv(index=False),
            file_name=f"api_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


def main() -> None:
    st.set_page_config(page_title="API Prediction Client", layout="wide")
    st.title("Prediction Gateway")
    st.caption("Streamlit frontend for FastAPI inference service")

    with st.sidebar:
        st.header("Connection")
        api_url = st.text_input("API Base URL", value=DEFAULT_API_URL)
        check = st.button("Check Health", use_container_width=True)
        if check:
            try:
                health = call_api(api_url, "/health")
                if health.ok:
                    st.success("API reachable")
                    st.json(health.json())
                else:
                    st.error(health.text)
            except Exception as exc:
                st.error(str(exc))

    tab_single, tab_batch = st.tabs(["Single", "Batch"])
    with tab_single:
        single_prediction_ui(api_url)
    with tab_batch:
        batch_prediction_ui(api_url)


if __name__ == "__main__":
    main()
