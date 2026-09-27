"""Stage 4: create the Docker Space if needed and push the deployment folder to it."""
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_building"))
from common import HF_TOKEN, ROOT, SPACE_REPO, USE_HUB  # noqa: E402


def main():
    if not USE_HUB:
        print("[local mode] No HF_TOKEN, so the Space was not deployed.")
        return
    create_repo(SPACE_REPO, repo_type="space", space_sdk="docker", token=HF_TOKEN, exist_ok=True, private=False)
    HfApi(token=HF_TOKEN).upload_folder(
        folder_path=str(ROOT / "deployment"), repo_id=SPACE_REPO, repo_type="space",
        commit_message="Deploy Streamlit app",
    )
    print(f"Deployed -> https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
