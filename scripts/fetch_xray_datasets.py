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
        # We use -nH and --cut-dirs to avoid deep nested folders from PhysioNet URL
        run_cmd(f'wget -r -N -c -np -nH --cut-dirs=4 '
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
            run_cmd(f'wget -c -O "{dest}" "{url}"')
        else:
            print(f"[+] {name} already exists. Skipping download.")
    
    # 4. Auto-Extraction
    print("\n--- PHASE 3: Extraction & Organization ---")
    
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
    print("\nNext Action: Run 'scripts/xray_landmarks_to_mask.py' to generate ground truth.")
    print("="*50)

if __name__ == "__main__":
    main()
