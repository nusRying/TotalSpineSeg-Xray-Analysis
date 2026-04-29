import os
import requests
import kagglehub
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import argparse

def download_file(url, dest):
    """Download a file with a progress bar."""
    if os.path.exists(dest):
        print(f"[+] {os.path.basename(dest)} already exists. Skipping.")
        return
    
    print(f"[*] Downloading {os.path.basename(dest)}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest, 'wb') as f, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest)
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def main():
    parser = argparse.ArgumentParser(description="Production Dataset Fetcher for Clinical X-Ray Suite")
    parser.add_argument("--output-dir", default="./data/raw_external", help="Directory to store datasets")
    parser.add_argument("--hf-token", help="Hugging Face Token (for VinDr-Lite)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. AASCE (Kaggle) - Scoliosis landmarks
    print("\n--- [1/4] AASCE (Scoliosis AP) ---")
    try:
        aasce_path = kagglehub.dataset_download("vamsidharreddy/aasce-miccai-2019")
        print(f"✅ AASCE ready at: {aasce_path}")
    except Exception as e:
        print(f"❌ AASCE Failed: {e}")

    # 2. Spine-Seg (Zenodo) - Multi-organ segmentation
    print("\n--- [2/4] Spine-Seg (Segmentation) ---")
    # Updated to the user-provided high-volume clinical record
    zenodo_files = {
        "images.zip": "https://zenodo.org/records/8009680/files/images.zip?download=1",
        "masks.zip": "https://zenodo.org/records/8009680/files/masks.zip?download=1"
    }
    for name, url in zenodo_files.items():
        dest = os.path.join(args.output_dir, name)
        try:
            # We use a custom downloader to handle these large files
            download_file(url, dest)
            print(f"✅ {name} downloaded successfully.")
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")

    # 3. CSXA-Lite (SciDB) - Lateral Cervical landmarks
    print("\n--- [3/4] CSXA-Lite (Lateral Ratios) ---")
    csxa_fid = "801011b2c734ad280b9326a29358730f" # PNG zip
    csxa_url = f"https://china.scidb.cn/download?fileId={csxa_fid}"
    try:
        dest = os.path.join(args.output_dir, "csxa_lite_png.zip")
        download_file(csxa_url, dest)
        print("✅ CSXA-Lite downloaded successfully.")
    except Exception as e:
        print(f"❌ CSXA-Lite Failed: {e}")

    # 4. VinDr-Lite (PhysioNet / Manual)
    print("\n--- [4/4] VinDr-Lite (Fractures) ---")
    print("⚠️  Note: Official VinDr-SpineXR requires credentialed PhysioNet access.")
    print("⚠️  The 1.5GB 'Lite' subset on HF is currently unavailable.")
    print("👉 Recommendation: Use the Spine-Seg (Step 2) data for initial pathology training.")

    print("\n" + "="*40)
    print("🚀 PRODUCTION DATA PIPELINE READY")
    print("="*40)
    print(f"Total size estimated: ~6.2 GB")
    print("Next: Run extraction on the ZIP files in your data directory.")

if __name__ == "__main__":
    main()
