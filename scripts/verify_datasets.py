import os
import kagglehub
from huggingface_hub import hf_hub_download
import requests
from tqdm import tqdm

def verify_aasce():
    print("\n--- Testing AASCE (Kaggle) ---")
    try:
        path = kagglehub.dataset_download("vamsidharreddy/aasce-miccai-2019")
        print(f"✅ AASCE Success! Downloaded to: {path}")
        return True
    except Exception as e:
        print(f"❌ AASCE Failed: {e}")
        return False

def verify_zenodo():
    print("\n--- Testing Spine-Seg (Zenodo) ---")
    url = "https://zenodo.org/record/6344265/files/manifest.json?download=1"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            print("✅ Zenodo Success! Connection established.")
            return True
    except Exception as e:
        print(f"❌ Zenodo Failed: {e}")
        return False

def verify_vindr(token):
    print("\n--- Testing VinDr (HuggingFace) ---")
    if not token or "YOUR_TOKEN" in token:
        print("⚠️ Skipping VinDr test: No valid HF token provided.")
        return False
    try:
        path = hf_hub_download(repo_id="v-sharma/vindr-spinexr", filename="README.md", token=token)
        print(f"✅ VinDr Success! Auth verified.")
        return True
    except Exception as e:
        print(f"❌ VinDr Failed: {e}")
        return False

if __name__ == "__main__":
    # Get token from environment variable or enter it here LOCALLY ONLY
    hf_token = os.getenv("HF_TOKEN", "YOUR_TOKEN_HERE")
    verify_aasce()
    verify_zenodo()
    verify_vindr(hf_token)
