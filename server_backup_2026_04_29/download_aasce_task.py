import kagglehub
import os
try:
    print('Starting Kaggle download...')
    path = kagglehub.dataset_download('vamsidharreddy/aasce-miccai-2019')
    target = '/workspace/data/raw_external/AASCE_2019'
    if not os.path.exists(target):
        os.symlink(path, target)
    print(f'SUCCESS: Dataset downloaded to {path} and linked to {target}')
except Exception as e:
    print(f'ERROR: {str(e)}')

