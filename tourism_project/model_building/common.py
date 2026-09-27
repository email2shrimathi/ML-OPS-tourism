"""Settings shared by every pipeline stage: repo IDs, paths, and the feature schema."""
import os
from pathlib import Path

HF_USER = os.getenv("HF_USER", "shrimathin")
DATASET_REPO = os.getenv("HF_DATASET_REPO", f"{HF_USER}/tourism-wellness-dataset")
MODEL_REPO = os.getenv("HF_MODEL_REPO", f"{HF_USER}/tourism-wellness-model")
SPACE_REPO = os.getenv("HF_SPACE_REPO", f"{HF_USER}/wellness-tourism-predictor")

HF_TOKEN = os.getenv("HF_TOKEN")
# Hub mode needs a token. LOCAL_ONLY=1 forces a fully offline run.
USE_HUB = bool(HF_TOKEN) and os.getenv("LOCAL_ONLY", "0") != "1"

ROOT = Path(__file__).resolve().parents[1]          # tourism_project/
RAW_CSV = ROOT / "data" / "tourism.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

TARGET = "ProdTaken"
NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_FILE = "wellness_model.joblib"
META_FILE = "model_metadata.json"
