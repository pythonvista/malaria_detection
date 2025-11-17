#!/usr/bin/env python3
"""
Streamlit version of the Malaria Symptom-Based Detection app.

Launch with:
    streamlit run streamlit_symptom_app.py
"""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import streamlit as st


MODEL_FILES = {
    "Decision Tree": "malaria_symptom_decision_tree.joblib",
    "SVM": "malaria_symptom_svm.joblib",
    "Logistic Regression": "malaria_symptom_logistic_regression.joblib",
    "Random Forest": "malaria_symptom_random_forest.joblib",
}

FEATURE_LABELS = {
    "age": "Patient Age (years)",
    "sex": "Gender",
    "fever": "Fever",
    "cold": "Cold / Chills",
    "rigor": "Rigor",
    "fatigue": "Fatigue",
    "headace": "Headache",
    "bitter_tongue": "Bitter Tongue",
    "vomitting": "Vomiting",
    "diarrhea": "Diarrhea",
    "Convulsion": "Convulsions",
    "Anemia": "Anemia",
    "jundice": "Jaundice",
    "cocacola_urine": "Coca-Cola Urine",
    "hypoglycemia": "Hypoglycemia",
    "prostraction": "Prostration",
    "hyperpyrexia": "Hyperpyrexia",
}


@st.cache_resource
def load_assets() -> Tuple[List[str], Dict[str, object], object]:
    base = Path(__file__).parent
    feature_path = base / "malaria_symptom_features.joblib"
    scaler_path = base / "malaria_symptom_scaler.joblib"

    if not feature_path.exists():
        return [], {}, None

    feature_names = joblib.load(feature_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    models = {}
    for model_name, filename in MODEL_FILES.items():
        model_path = base / filename
        if model_path.exists():
            models[model_name] = joblib.load(model_path)

    return feature_names, models, scaler


def collect_inputs(feature_names: List[str]) -> Dict[str, float]:
    values: Dict[str, float] = {}

    st.subheader("Patient Demographics")
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    sex = st.selectbox("Gender", options=["Female", "Male"], index=0)

    values["age"] = float(age)
    values["sex"] = 1.0 if sex == "Male" else 0.0

    st.subheader("Clinical Symptoms")
    cols = st.columns(2)
    for idx, feature in enumerate(feature_names):
        if feature in ("age", "sex"):
            continue
        label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
        col = cols[idx % 2]
        choice = col.radio(
            label,
            options=["No", "Yes"],
            horizontal=True,
            key=f"symptom_{feature}",
        )
        values[feature] = 1.0 if choice == "Yes" else 0.0

    return values


def predict(model, scaler, features: List[str], values: Dict[str, float], scale: bool):
    vector = np.array([values[f] for f in features]).reshape(1, -1)
    if scale and scaler is not None:
        vector = scaler.transform(vector)
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    return prediction, probabilities


def main():
    st.set_page_config(page_title="Malaria Symptom Detection", page_icon="🦠", layout="wide")
    st.title("🦠 Malaria Symptom-Based Detection (Streamlit)")

    feature_names, models, scaler = load_assets()
    if not feature_names or not models:
        st.error("Required model artifacts are missing. Please run the training pipeline first.")
        return

    with st.sidebar:
        st.header("Quick Actions")
        st.markdown("- Load models from local joblib artifacts\n- Choose any algorithm\n- View probability scores")
        sample = st.button("Load Severe Malaria Sample")

    with st.form("symptom_form"):
        inputs = collect_inputs(feature_names)
        if sample:
            for feature in feature_names:
                if feature == "age":
                    inputs[feature] = 25.0
                elif feature == "sex":
                    inputs[feature] = 1.0
                else:
                    inputs[feature] = 1.0

        selected_model = st.selectbox("Select Model", options=list(models.keys()), index=3)
        submitted = st.form_submit_button("Analyze Symptoms")

    if submitted:
        model = models[selected_model]
        needs_scaler = selected_model in {"SVM", "Logistic Regression"}
        prediction, probabilities = predict(model, scaler, feature_names, inputs, needs_scaler)

        st.subheader("Prediction Result")
        severe_prob = probabilities[1]
        no_severe_prob = probabilities[0]

        if prediction == 1:
            st.error(f"Severe malaria risk detected ({severe_prob:.1%}). Immediate evaluation recommended.")
        else:
            st.success(f"Low malaria risk detected ({no_severe_prob:.1%}). Continue monitoring.")

        st.metric("Probability - Severe Malaria", f"{severe_prob:.1%}")
        st.metric("Probability - No Severe Malaria", f"{no_severe_prob:.1%}")

        with st.expander("Symptom Summary"):
            positives = [FEATURE_LABELS.get(f, f.replace('_', ' ').title()) for f in feature_names if f not in ("age", "sex") and inputs[f] == 1.0]
            if positives:
                st.write(", ".join(positives))
            else:
                st.write("No symptoms selected.")

        st.caption("Deploy with `streamlit run streamlit_symptom_app.py` and share the provided URL for remote access.")


if __name__ == "__main__":
    main()

