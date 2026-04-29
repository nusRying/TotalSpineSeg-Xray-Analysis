import os
import shutil
import json
import glob

def prepare_nnunet_dataset():
    base_dir = "/workspace/data/raw_external/Mendeley_Lumbar_Seg/v2_raw"
    target_base = "/workspace/nnUNet_raw/Dataset250_MendeleyLumbar"
    
    images_tr = os.path.join(target_base, "imagesTr")
    labels_tr = os.path.join(target_base, "labelsTr")
    images_ts = os.path.join(target_base, "imagesTs")
    
    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)
    os.makedirs(images_ts, exist_ok=True)
    
    # Process Train, Valid, Test
    # We will combine train and valid for training, and use test for imagesTs
    
    sets = [('train', True), ('valid', True), ('test', False)]
    
    count = 0
    for set_name, is_train in sets:
        path = os.path.join(base_dir, set_name)
        # Find all jpg files
        images = glob.glob(os.path.join(path, "*.jpg"))
        for img_path in images:
            case_name = os.path.basename(img_path).replace(".jpg", "")
            mask_path = img_path.replace(".jpg", "_mask.png")
            
            if is_train:
                # nnU-Net expects images as case_name_0000.png (or .jpg)
                shutil.copy(img_path, os.path.join(images_tr, f"{case_name}_0000.jpg"))
                if os.path.exists(mask_path):
                    shutil.copy(mask_path, os.path.join(labels_tr, f"{case_name}.png"))
            else:
                shutil.copy(img_path, os.path.join(images_ts, f"{case_name}_0000.jpg"))
            count += 1
            
    print(f"Processed {count} images into nnU-Net format.")

    # Create dataset.json
    dataset_json = {
        "channel_names": {
            "0": "RGB"
        },
        "labels": {
            "background": 0,
            "L1": 1,
            "L2": 2,
            "L3": 3,
            "L4": 4,
            "L5": 5
        },
        "numTrainingCases": 0, # Will fill this
        "file_ending": ".jpg",
        "overwrite_image_reader_writer": "NaturalImage2DIO" # Since these are RGB JPEGs
    }
    
    # Update numTrainingCases
    dataset_json["numTrainingCases"] = len(glob.glob(os.path.join(images_tr, "*_0000.jpg")))
    
    with open(os.path.join(target_base, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)
        
    print("dataset.json created.")

if __name__ == "__main__":
    prepare_nnunet_dataset()
