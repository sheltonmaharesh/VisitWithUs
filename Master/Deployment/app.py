import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download, login
from huggingface_hub.utils import HfHubHTTPError


# ----------------------------
# Basic config
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Hugging Face cache dirs (HF Spaces friendly)
cache_dir = "/tmp/hf_cache"
os.environ.setdefault("HF_HOME", cache_dir)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
os.makedirs(cache_dir, exist_ok=True)


# ----------------------------
# App config
# ----------------------------
@dataclass
class AppConfig:
    model_repo_id: str = "sheltonmaharesh/Tourism_Prediction_Model"
    repo_type: str = "model"
    # Use generic filenames so you don't hardcode a specific winning model name
    model_filename: str = "Model_Dump_JOBLIB/best_model.joblib"
    threshold_filename: str = "Model_Dump_JOBLIB/best_threshold.txt"
    hf_token_env: str = "HUGGINGFACE_TOKEN"


CFG = AppConfig()


# ----------------------------
# Helpers
# ----------------------------
def maybe_login() -> None:
    """
    Login only if a token exists.
    Public repos work without login; private repos require it.
    """
    token = os.getenv(CFG.hf_token_env)
    if token:
        try:
            login(token=token)
            logger.info("Logged in to Hugging Face.")
        except Exception as e:
            logger.warning("HF login failed (continuing anyway): %s", e)


@st.cache_resource(show_spinner=True)
def load_model_and_threshold() -> Tuple[Any, float]:
    """
    Downloads and loads the model + threshold once per app session.
    """
    maybe_login()

    try:
        model_path = hf_hub_download(
            repo_id=CFG.model_repo_id,
            filename=CFG.model_filename,
            repo_type=CFG.repo_type,
            token=os.getenv(CFG.hf_token_env),
        )
        threshold_path = hf_hub_download(
            repo_id=CFG.model_repo_id,
            filename=CFG.threshold_filename,
            repo_type=CFG.repo_type,
            token=os.getenv(CFG.hf_token_env),
        )

        model = joblib.load(model_path)
        with open(threshold_path, "r") as f:
            threshold = float(f.read().strip())

        return model, threshold

    except FileNotFoundError as e:
        raise RuntimeError(f"Missing model artifacts in repo: {e}") from e
    except HfHubHTTPError as e:
        raise RuntimeError(f"Hugging Face download error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load model/threshold: {e}") from e


def normalize_inputs(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize user inputs to match training data categories as closely as possible.
    This prevents category mismatch from breaking one-hot features.
    """
    # Fix common category spelling/variants
    if "TypeofContact" in d:
        d["TypeofContact"] = (
            "Self Inquiry" if d["TypeofContact"] in ["Self Enquiry", "Self Inquiry"] else "Company Invited"
        )

    if "Occupation" in d:
        mapping = {
            "Free Lancer": "Freelancer",
            "Freelancer": "Freelancer",
        }
        d["Occupation"] = mapping.get(d["Occupation"], d["Occupation"])

    if "MaritalStatus" in d:
        # If your training data used "Single/Married/Divorced", push inputs into those buckets.
        if d["MaritalStatus"] == "Unmarried":
            d["MaritalStatus"] = "Single"

    return d


def predict(model: Any, threshold: float, data: Dict[str, Any]) -> Tuple[int, float]:
    df = pd.DataFrame([data])
    prob = float(model.predict_proba(df)[:, 1][0])
    pred = int(prob >= threshold)
    return pred, prob


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Visit With Us - Tourism Package Prediction", layout="wide")

st.title("Visit With Us: Tourism Package Prediction")
st.write("Enter customer details to predict the likelihood of purchasing the tourism package.")

# Load model safely
try:
    with st.spinner("Loading model..."):
        model, best_threshold = load_model_and_threshold()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(str(e))
    st.stop()

with st.form("customer_form"):
    st.header("Customer Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=41)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("MaritalStatus", ["Married", "Single", "Divorced", "Unmarried"])
        occupation = st.selectbox("Occupation", ["Freelancer", "Free Lancer", "Salaried", "Small Business", "Large Business"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("MonthlyIncome", min_value=0, max_value=1_000_000, value=20999)

    with col2:
        typeofcontact = st.selectbox("TypeofContact", ["Self Inquiry", "Self Enquiry", "Company Invited"])
        citytier = st.selectbox("CityTier", [1, 2, 3], index=2)
        duration_of_pitch = st.number_input("DurationOfPitch", min_value=1, max_value=60, value=6)
        product_pitched = st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        preferred_property_star = st.selectbox("PreferredPropertyStar", [1, 2, 3, 4, 5], index=2)
        number_of_trips = st.number_input("NumberOfTrips", min_value=0, max_value=30, value=1)

    with col3:
        number_of_person_visiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=3)
        number_of_followups = st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=3)
        number_of_children_visiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=5, value=0)
        passport = st.selectbox("Passport", ["Yes", "No"])
        own_car = st.selectbox("OwnCar", ["Yes", "No"])
        pitch_satisfaction_score = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=3)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        "Age": age,
        "TypeofContact": typeofcontact,
        "CityTier": citytier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }

    input_data = normalize_inputs(input_data)

    try:
        pred, prob = predict(model, best_threshold, input_data)

        st.subheader("Prediction Result")
        st.write(f"**Probability of purchase:** {prob:.3f}")
        st.write(f"**Threshold:** {best_threshold:.3f}")
        st.write("**Decision:** " + ("Likely to purchase ✅" if pred == 1 else "Unlikely to purchase ❌"))

    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        st.error(f"Prediction failed: {e}")
