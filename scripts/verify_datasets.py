import os
import kagglehub
from huggingface_hub import hf_hub_download
import requests
from tqdm import tqdm

def verify_aasce():
    print("\n--- Testing AASCE (Kaggle) ---")
    # Ensure credentials from environment are used
    if "KAGGLE_USERNAME" not in os.environ:
        os.environ["KAGGLE_USERNAME"] = "nusrying"
        os.environ["KAGGLE_KEY"] = "677603aff6d544733c4cb8eb22b32cea"
        
    try:
        path = kagglehub.dataset_download("vamsidharreddy/aasce-miccai-2019")
        print(f"[OK] AASCE Success! Downloaded to: {path}")
        return True
    except Exception as e:
        print(f"[FAIL] AASCE Failed: {e}")
        return False

def verify_zenodo():
    print("\n--- Testing Spine-Seg (Zenodo) ---")
    url = "https://zenodo.org/record/6344265/files/manifest.json?download=1"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            print("[OK] Zenodo Success! Connection established.")
            return True
    except Exception as e:
        print(f"[FAIL] Zenodo Failed: {e}")
        return False

def verify_csxa():
    print("\n--- Testing CSXA-Lite (SciDB) ---")
    fid = "801011b2c734ad280b9326a29358730f"
    url = f"https://china.scidb.cn/download?fileId={fid}"
    try:
        r = requests.head(url, timeout=10)
        if r.status_code in [200, 302]:
            print("[OK] CSXA-Lite Verified (Connection OK)")
            return True
        else:
            print(f"[FAIL] CSXA-Lite Failed: Status {r.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] CSXA-Lite Failed: {e}")
        return False

def verify_vindr(token):
    print("\n--- Testing VinDr-Lite (HuggingFace) ---")
    if not token or "YOUR_TOKEN" in token:
        print("[!] Skipping VinDr-Lite test: No valid HF token provided.")
        return False
    try:
        path = hf_hub_download(repo_id="v-sharma/vindr-spinexr", filename="README.md", token=token, repo_type="dataset")
        print(f"[OK] VinDr-Lite Verified (Auth OK)")
        return True
    except Exception as e:
        print(f"[FAIL] VinDr-Lite Failed: {e}")
        return False

if __name__ == "__main__":
    # Get token from environment variable or enter it here LOCALLY ONLY
    hf_token = os.getenv("HF_TOKEN", "YOUR_TOKEN_HERE")
    
    results = {
        "AASCE": verify_aasce(),
        "Spine-Seg": verify_zenodo(),
        "CSXA-Lite": verify_csxa(),
        "VinDr-Lite": verify_vindr(hf_token)
    }
    
    print("\n" + "="*30)
    print("  FINAL VERIFICATION SUMMARY")
    print("="*30)
    for ds, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{ds:12}: {icon}")
    print("="*30)
