"""Stage 2: clean the raw data, create stratified train/test splits, and publish them."""
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

from common import (DATASET_REPO, FEATURES, HF_TOKEN, PROCESSED_DIR, RAW_CSV, TARGET, USE_HUB)

CAPS = {"DurationOfPitch": 60, "NumberOfTrips": 12}     # upper limits found during EDA


def load_raw() -> pd.DataFrame:
    if USE_HUB:
        path = hf_hub_download(DATASET_REPO, "raw/tourism.csv", repo_type="dataset", token=HF_TOKEN)
        return pd.read_csv(path)
    return pd.read_csv(RAW_CSV)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")] + ["CustomerID"], errors="ignore")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    # Designation maps 1:1 onto ProductPitched, so it adds nothing
    df = df.drop(columns=["Designation"], errors="ignore")
    for col, cap in CAPS.items():
        df[col] = df[col].clip(upper=cap)
    before = len(df)
    df = df.drop_duplicates().dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate/invalid rows")
    return df[FEATURES + [TARGET]]


def main():
    df = clean(load_raw())
    print(f"Clean data: {df.shape}, positive rate = {df[TARGET].mean():.3f}")
    train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=42)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"train={train.shape}  test={test.shape}")

    if USE_HUB:
        HfApi(token=HF_TOKEN).upload_folder(
            folder_path=str(PROCESSED_DIR), path_in_repo="processed",
            repo_id=DATASET_REPO, repo_type="dataset", commit_message="Update processed splits",
        )
        print(f"Uploaded splits -> https://huggingface.co/datasets/{DATASET_REPO}/tree/main/processed")


if __name__ == "__main__":
    main()
