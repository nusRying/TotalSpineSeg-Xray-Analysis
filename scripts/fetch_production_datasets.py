import os
import requests
import kagglehub
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
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                size = f.write(chunk)
                bar.update(size)

def main():
    parser = argparse.ArgumentParser(description="Production Dataset Fetcher for Clinical X-Ray Suite")
    parser.add_argument("--output-dir", default="./data/raw_external", help="Directory to store datasets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Configure Credentials (nusrying credentials provided by user)
    os.environ['KAGGLE_USERNAME'] = "nusrying"
    os.environ['KAGGLE_KEY'] = "677603aff6d544733c4cb8eb22b32cea"

    # 1. AASCE (Kaggle Mirror) - Using the public mirror to avoid 403 errors
    print("\n--- [1/3] AASCE (Kaggle Mirror) ---")
    try:
        # Using the mirror found to be working on Colab
        aasce_path = kagglehub.dataset_download("wahyurahmaniar/aasce-miccai-2019-x-ray-dataset")
        print(f"✅ AASCE downloaded to: {aasce_path}")
    except Exception as e:
        print(f"❌ AASCE Error: {e}")

    # 2. SPINE-SEG (Consolidated Clinical Data)
    print("\n--- [2/3] Spine-Seg (Zenodo Archive) ---")
    spine_seg_url = "https://zenodo.org/api/records/8009680/files-archive"
    try:
        dest = os.path.join(args.output_dir, "spine_seg_archive.zip")
        download_file(spine_seg_url, dest)
        print("✅ Spine-Seg archive downloaded successfully.")
    except Exception as e:
        print(f"❌ Failed to download Spine-Seg: {e}")

    # 3. CSXA (Lateral Ratios)
    print("\n--- [3/3] CSXA (SciDB) ---")
    csxa_url = "https://china.scidb.cn/download?fileId=801011b2c734ad280b9326a29358730f"
    try:
        dest = os.path.join(args.output_dir, "csxa_lite_png.zip")
        download_file(csxa_url, dest)
        print("✅ CSXA-Lite downloaded successfully.")
    except Exception as e:
        print(f"❌ CSXA-Lite Failed: {e}")

    print("\n" + "="*40)
    print("🏁 PRODUCTION DATAREADY")
    print(f"Location: {args.output_dir}")
    print("="*40)

if __name__ == "__main__":
    main()

