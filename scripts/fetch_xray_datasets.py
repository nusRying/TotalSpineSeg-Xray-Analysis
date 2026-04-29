import os
import subprocess
import argparse
import sys

def run_cmd(cmd):
    """Run a shell command and print output."""
    print(f">> Executing: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!! Command failed with exit code {e.returncode}")
        # We don't exit here to allow resuming other parts of the script
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="TotalSpineSeg: Ultimate X-Ray Dataset Downloader")
    parser.add_argument("--output-dir", required=True, help="Base directory to store datasets (e.g., ./data)")
    parser.add_argument("--user", required=True, help="PhysioNet Username for VinDr-SpineXR")
    parser.add_argument("--password", required=True, help="PhysioNet Password for VinDr-SpineXR")
    parser.add_argument("--scidb-user", help="SciDB Email (optional if open)")
    parser.add_argument("--scidb-pass", help="SciDB Password (optional if open)")
    parser.add_argument("--skip-vindr", action="store_true", help="Skip the 36GB VinDr download")
    args = parser.parse_args()

    # 1. Create Folder Structure
    base_raw = os.path.join(args.output_dir, "raw_external")
    paths = {
        "vindr": os.path.join(base_raw, "VinDr_SpineXR"),
        "csxa": os.path.join(base_raw, "CSXA_V3")
    }
    
    print(f"[*] Initializing workspace in {args.output_dir}...")
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    # 2. Download VinDr-SpineXR (~36GB)
    if not args.skip_vindr:
        print("\n--- PHASE 1: VinDr-SpineXR (Large Download: ~36GB) ---")
        # -r: recursive, -N: only get newer, -c: continue/resume, -np: no-parent
        # --reject "index.html*" avoids downloading the directory listing files
        run_cmd(f'wget -r -N -c -np -nH --cut-dirs=4 --reject "index.html*" '
                f'--user {args.user} --password "{args.password}" '
                f'-P {paths["vindr"]} https://physionet.org/files/vindr-spinexr/1.0.0/')
    else:
        print("\n[!] Skipping VinDr-SpineXR as requested.")

    # 3. Download CSXA V3.0 (~3GB) From SciDB
    print("\n--- PHASE 2: CSXA V3.0 (Cervical Lateral: ~3GB) ---")
    csxa_files = {
        "images.zip": "801011b2c734ad280b9326a29358730f",
        "labels.zip": "5dada884dd8d622531e826f2452e35d7"
    }
    for name, fid in csxa_files.items():
        url = f"https://china.scidb.cn/download?fileId={fid}"
        dest = os.path.join(paths["csxa"], name)
        if not os.path.exists(dest):
            print(f"[*] Downloading {name}...")
            # If SciDB credentials are provided, try to use them (basic auth support)
            auth_str = ""
            if args.scidb_user and args.scidb_pass:
                auth_str = f'--user {args.scidb_user} --password "{args.scidb_pass}" '
            
            run_cmd(f'wget -c {auth_str}-O "{dest}" "{url}"')
        else:
            print(f"[+] {name} already exists. Skipping download.")
    
    # 4. Download AASCE (MICCAI 2019) From Kaggle
    print("\n--- PHASE 3: AASCE 2019 (Scoliosis AP: ~1GB) ---")
    try:
        import kagglehub
        # Downloads to ~/.cache/kagglehub/datasets/vamsidharreddy/aasce-miccai-2019/
        aasce_path = kagglehub.dataset_download("vamsidharreddy/aasce-miccai-2019")
        print(f"[*] AASCE downloaded to: {aasce_path}")
        # Symlink or move it to our target dir
        target_aasce = os.path.join(base_raw, "AASCE_2019")
        if not os.path.exists(target_aasce):
            # Use os.symlink for cleaner structure if on Linux (server)
            run_cmd(f'ln -s "{aasce_path}" "{target_aasce}"')
    except Exception as e:
        print(f"[!] Kaggle download failed: {e}")
        print("[!] Manual download required for AASCE if not in cache.")

    # 5. Auto-Extraction
    print("\n--- PHASE 4: Extraction & Organization ---")
    
    # CSXA Extraction
    img_zip = os.path.join(paths["csxa"], "images.zip")
    ann_zip = os.path.join(paths["csxa"], "labels.zip")
    
    if os.path.exists(img_zip):
        run_cmd(f'unzip -qo "{img_zip}" -d "{os.path.join(paths["csxa"], "images")}"')
    if os.path.exists(ann_zip):
        run_cmd(f'unzip -qo "{ann_zip}" -d "{os.path.join(paths["csxa"], "annotations")}"')

    print("\n" + "="*50)
    print("✅ DATASET FETCH COMPLETE")
    print("="*50)
    print(f"VinDr Path: {paths['vindr']}")
    print(f"CSXA Path:  {paths['csxa']}")
    print(f"AASCE Path: {os.path.join(base_raw, 'AASCE_2019')}")
    print("\nNext Action: Run 'scripts/xray_landmarks_to_mask.py' to generate ground truth.")
    print("="*50)

if __name__ == "__main__":
    main()
