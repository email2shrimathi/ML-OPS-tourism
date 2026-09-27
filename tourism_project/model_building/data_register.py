"""Stage 1: publish the raw CSV to a versioned Hugging Face dataset repo."""
from huggingface_hub import HfApi, create_repo

from common import DATASET_REPO, HF_TOKEN, RAW_CSV, USE_HUB


def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"{RAW_CSV} is missing")
    if not USE_HUB:
        print(f"[local mode] Skipping upload. Raw data stays at {RAW_CSV}")
        return
    create_repo(DATASET_REPO, repo_type="dataset", token=HF_TOKEN, exist_ok=True, private=False)
    HfApi(token=HF_TOKEN).upload_file(
        path_or_fileobj=str(RAW_CSV), path_in_repo="raw/tourism.csv",
        repo_id=DATASET_REPO, repo_type="dataset", commit_message="Register raw tourism data",
    )
    print(f"Registered raw data -> https://huggingface.co/datasets/{DATASET_REPO}")


if __name__ == "__main__":
    main()
