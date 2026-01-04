import os
import logging
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tourism-app")

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Visit With Us: Tourism Package Prediction", layout="wide")
st.title("Visit With Us: Tourism Package Prediction")
st.write("Enter customer details to predict the likelihood of purchasing the tourism package.")

# -----------------------------
# Hugging Face configuration
# -----------------------------
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "sheltonmaharesh/Tourism_Prediction_Model")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")  # support both

CACHE_DIR = "/tmp/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR


# -----------------------------
# Utilities
# -----------------------------
def _discover_best_model_file(repo_id: str, token: str | None) -> str:
    """
    Find the best model artifact in HF model repo.
    Expected pattern: Model_Dump_JOBLIB/BestModel_*.joblib
    """
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")

    candidates = sorted(
        [
            f for f in files
            if f.startswith("Model_Dump_JOBLIB/BestModel_") and f.endswith(".joblib")
        ]
    )
    if not candidates:
        raise FileNotFoundError(
            f"No BestModel_*.joblib found in {repo_id}. Found files: {files}"
        )
    return candidates[-1]


@st.cache_resource(show_spinner=True)
def load_model_and_threshold(repo_id: str, token: str | None):
    """
    Loads the best model + best threshold from Hugging Face.
    Cached so Streamlit doesn't download on every interaction.
    """
    best_model_file = _discover_best_model_file(repo_id, token)
    logger.info("Best model file discovered: %s", best_model_file)

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=best_model_file,
        repo_type="model",
        token=token,
        cache_dir=CACHE_DIR,
    )

    threshold_path = hf_hub_download(
        repo_id=repo_id,
        filename="Model_Dump_JOBLIB/best_threshold.txt",
        repo_type="model",
        token=token,
        cache_dir=CACHE_DIR,
    )

    model = joblib.load(model_path)

    with open(threshold_path, "r") as f:
        best_threshold = float(f.read().strip())

    return model, best_threshold, best_model_file


def predict_purchase(model, best_threshold: float, input_dict: dict) -> tuple[int, float]:
    """
    Returns (prediction, probability).
    """
    df = pd.DataFrame([input_dict])
    prob = float(model.predict_proba(df)[:, 1][0])
    pred = int(prob >= best_threshold)
    return pred, prob


# -----------------------------
# Load model
# -----------------------------
if not HF_TOKEN:
    st.error(
        "Missing Hugging Face token. Set `HUGGINGFACE_TOKEN` (recommended) "
        "or `HF_TOKEN` in the Space secrets/environment."
    )
    st.stop()

try:
    model, best_threshold, best_model_file = load_model_and_threshold(HF_MODEL_REPO, HF_TOKEN)
    st.success(f"Loaded model: {best_model_file} (threshold={best_threshold:.4f})")
except Exception as ex:
    st.error(f"Hugging Face download/load error: {ex}")
    st.stop()


# -----------------------------
# Input form (match training schema)
# -----------------------------
with st.form("customer_form"):
    st.subheader("Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=41)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        MaritalStatus = st.selectbox("MaritalStatus", ["Married", "Unmarried", "Single", "Divorced"])
        Occupation = st.selectbox("Occupation", ["Free Lancer", "Salaried", "Small Business", "Large Business"])
        Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        MonthlyIncome = st.number_input("MonthlyIncome", min_value=0, max_value=1_000_000, value=20999)

    with col2:
        TypeofContact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
        CityTier = st.selectbox("CityTier", [1, 2, 3], index=2)
        DurationOfPitch = st.number_input("DurationOfPitch", min_value=1, max_value=60, value=6)
        ProductPitched = st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        PreferredPropertyStar = st.selectbox("PreferredPropertyStar", [1, 2, 3, 4, 5], index=2)
        NumberOfTrips = st.number_input("NumberOfTrips", min_value=0, max_value=30, value=1)

    with col3:
        NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=3)
        NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=3)
        NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=5, value=0)
        Passport = st.selectbox("Passport", ["Yes", "No"], index=1)
        OwnCar = st.selectbox("OwnCar", ["Yes", "No"], index=1)
        PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=3)

    submitted = st.form_submit_button("Predict")


# -----------------------------
# Predict
# -----------------------------
if submitted:
    # Convert to the exact feature schema expected by the pipeline
    input_data = {
        "Age": int(Age),
        "TypeofContact": TypeofContact,
        "CityTier": int(CityTier),
        "DurationOfPitch": int(DurationOfPitch),
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": int(NumberOfPersonVisiting),
        "NumberOfFollowups": int(NumberOfFollowups),
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": int(PreferredPropertyStar),
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": int(NumberOfTrips),
        "Passport": 1 if Passport == "Yes" else 0,
        "OwnCar": 1 if OwnCar == "Yes" else 0,
        "PitchSatisfactionScore": int(PitchSatisfactionScore),
        "NumberOfChildrenVisiting": int(NumberOfChildrenVisiting),
        "Designation": Designation,
        "MonthlyIncome": float(MonthlyIncome),
    }

    try:
        pred, prob = predict_purchase(model, best_threshold, input_data)

        st.subheader("Prediction")
        st.write(f"**Probability of purchase:** {prob:.4f}")
        if pred == 1:
            st.success("Likely to purchase the tourism package ✅")
        else:
            st.warning("Unlikely to purchase the tourism package ❌")

        with st.expander("Show input payload (debug)"):
            st.json(input_data)

    except Exception as ex:
        logger.exception("Prediction failed")
        st.error(f"Prediction error: {ex}")
