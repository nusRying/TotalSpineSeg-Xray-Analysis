import kagglehub
import shutil
from pathlib import Path

def download_verse():
    print("Downloading VerSe 2019 dataset...")
    # Using a known VerSe 2019 dataset on Kaggle
    path = kagglehub.dataset_download("vlbscat/verse2019")
    print(f"Dataset downloaded to: {path}")
    
    target_dir = Path("data/raw/verse2019")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # We only need a sample for DRR generation to prove the concept
    print(f"Moving files to {target_dir}...")
    # The structure of the downloaded folder varies, so we copy everything
    # or look for .nii.gz files
    src_path = Path(path)
    for file in src_path.rglob("*.nii.gz"):
        dest = target_dir / file.name
        if not dest.exists():
            shutil.copy(file, dest)
            print(f"Copied {file.name}")

if __name__ == "__main__":
    download_verse()
