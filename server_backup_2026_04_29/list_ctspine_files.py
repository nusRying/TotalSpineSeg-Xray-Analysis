import sys
sys.path.insert(0, '/workspace')
from huggingface_hub import HfApi

api = HfApi()
files = list(api.list_repo_files(repo_id="alexanderdann/CTSpine1K", repo_type="dataset"))
print("Total files:", len(files))
for f in files[:30]:
    print(f)
