import os
import subprocess
from pathlib import Path

def check_status(name, condition, details="", action=""):
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"{status} | {name.ljust(25)} | {details}")
    if not condition and action:
        print(f"      👉 ACTION: {action}")
    return condition

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
    except:
        return None

def main():
    print("\n" + "="*70)
    print("      TOTALSPINESEG X-RAY: SERVER READINESS REPORT")
    print("="*70 + "\n")
    
    root = Path(__file__).resolve().parents[1]
    all_pass = True

    # 1. Hardware Check
    gpu = run_cmd("nvidia-smi --query-gpu=name --format=csv,noheader")
    all_pass &= check_status("NVIDIA GPU", gpu is not None, gpu or "No GPU found", "Ensure NVIDIA drivers are installed.")

    # 2. Docker Check
    docker_v = run_cmd("docker --version")
    all_pass &= check_status("Docker Installed", docker_v is not None, docker_v or "Docker missing", "Install Docker Engine.")
    
    docker_gpu = run_cmd("docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi")
    all_pass &= check_status("Docker GPU Link", docker_gpu is not None, "NVIDIA-container-runtime active" if docker_gpu else "Link failed", "Install nvidia-container-toolkit.")

    # 3. Weights Check
    weights_dir = root / "weights" / "Dataset202_TotalSpineSeg_XRay_FullSpine_AP_Lat" / "nnUNetTrainer__nnUNetPlans__2d" / "fold_0"
    weights_path = weights_dir / "checkpoint_final.pth"
    all_pass &= check_status("Model Weights", weights_path.exists(), 
                            f"Found in fold_0" if weights_path.exists() else "Missing checkpoint_final.pth",
                            f"Ensure weights are at: weights/Dataset202.../nnUNetTrainer__nnUNetPlans__2d/fold_0/")

    # 4. Data Check
    img_dir = root / "data" / "xray" / "images"
    mask_dir = root / "data" / "xray" / "masks"
    
    imgs = list(img_dir.glob("*.png")) if img_dir.exists() else []
    masks = list(mask_dir.glob("*.png")) if mask_dir.exists() else []
    
    data_exists = len(imgs) > 0 and len(masks) > 0
    match = len(imgs) == len(masks) and data_exists
    
    all_pass &= check_status("Training Data", data_exists, f"{len(imgs)} images, {len(masks)} masks found", 
                            f"Upload PNG files to data/xray/images and data/xray/masks.")
    all_pass &= check_status("Data Consistency", match, "Counts match" if match else f"mismatch: {len(imgs)} images vs {len(masks)} masks",
                            "Ensure every image has a corresponding mask with the exact same name.")

    # 5. Docker Image Check
    img_exists = run_cmd("docker images -q totalspineseg-milestone4")
    all_pass &= check_status("Project Container", img_exists is not None and len(img_exists) > 0, 
                            "totalspineseg-milestone4 built", "Run: docker build -t totalspineseg-milestone4 .")

    print("\n" + "="*70)
    if all_pass:
        print("🚀 RESULT: SYSTEM IS READY FOR HIGH-ACCURACY TRAINING!")
        print("\nNEXT STEPS:")
        print("1. Start tmux:  tmux new -s training")
        print("2. Run build:   docker build -t totalspineseg-a100 .")
        print("3. Run training: docker run --gpus all -v $(pwd):/app totalspineseg-a100 nnUNetv2_train 202 2d 0")
    else:
        print("⚠️  RESULT: SYSTEM NOT READY.")
        print("Please resolve the ACTION items above.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
