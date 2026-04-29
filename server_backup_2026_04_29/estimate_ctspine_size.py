import sys
import os
sys.path.insert(0, '/workspace')
from huggingface_hub import HfApi, hf_hub_url

api = HfApi()

# Get all files
all_files = list(api.list_repo_files(repo_id="alexanderdann/CTSpine1K", repo_type="dataset"))
total = len(all_files)

# Filter only the NIfTI raw data files (volumes + labels)
nifti_files = [f for f in all_files if f.startswith("raw_data/") and f.endswith(".nii.gz")]
print(f"Total files in repo: {total}")
print(f"NIfTI raw data files: {len(nifti_files)}")

# Count sizes by fetching file info for first 5 to estimate average
sample_files = api.list_repo_tree(repo_id="alexanderdann/CTSpine1K", repo_type="dataset", path_in_repo="raw_data/labels/COLONOG")
for item in list(sample_files)[:5]:
    print(f"  {item.path}: {item.size / 1024 / 1024:.1f} MB")
