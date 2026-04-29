import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_file(args):
    url, dest, user, password = args
    if os.path.exists(dest):
        return True
    
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, auth=(user, password), headers=headers, stream=True, timeout=30)
        if response.status_code == 200:
            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            return False
    except Exception:
        return False

def main():
    user = "umairejaz04"
    password = "Umair@825"
    base_url = "https://physionet.org/files/vindr-spinexr/1.0.0/"
    dest_root = "/workspace/data/raw_external/VinDr_SpineXR"
    sums_file = "/workspace/data/raw_external/VinDr_Kaggle_Subset/physionet.org/files/vindr-spinexr/1.0.0/SHA256SUMS.txt"
    
    files_to_download = []
    with open(sums_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                file_path = parts[1]
                if file_path.endswith('.dicom'):
                    files_to_download.append(file_path)
    
    print(f"Total files to check: {len(files_to_download)}")
    
    tasks = []
    for f in files_to_download:
        url = base_url + f
        dest = os.path.join(dest_root, f)
        tasks.append((url, dest, user, password))
    
    # Use 16 threads for parallel download
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(download_file, tasks), total=len(tasks)))

if __name__ == "__main__":
    main()
