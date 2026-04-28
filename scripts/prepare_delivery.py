import os
import shutil
import json
from pathlib import Path

def main():
    print("\n" + "="*60)
    print("      Preparing Milestone 4 Final Delivery Package")
    print("="*60 + "\n")
    
    root = Path(__file__).resolve().parents[1]
    delivery_dir = root / "Milestone4_Delivery_Package"
    delivery_dir.mkdir(exist_ok=True)

    # 1. Gather Weights
    # Note: We find the most recent training folder
    results_root = root / "data" / "nnUNet" / "results"
    try:
        # Find the specific Dataset202 folder
        model_dir = next(results_root.glob("Dataset202*"))
        trainer_dir = next(model_dir.glob("nnUNetTrainer*"))
        fold_dir = trainer_dir / "fold_0"
        
        # Copy best checkpoint
        shutil.copy(fold_dir / "checkpoint_best.pth", delivery_dir / "checkpoint_final.pth")
        # Copy log for "Learning Stats"
        log_file = sorted(fold_dir.glob("training_log_*.txt"))[-1]
        shutil.copy(log_file, delivery_dir / "learning_stats.txt")
        print("✅ Successfully gathered Best Weights and Learning Stats.")
    except Exception as e:
        print(f"⚠️  Warning: Could not find training results. Error: {e}")

    # 2. Gather Configuration
    shutil.copy(root / "Dockerfile", delivery_dir / "Dockerfile")
    shutil.copy(root / "requirements.txt", delivery_dir / "requirements.txt")
    shutil.copy(root / "README_DOCKER.md", delivery_dir / "README_DOCKER.md")
    print("✅ Successfully gathered Environment Configs.")

    # 3. Gather Strategic Docs
    docs_dir = delivery_dir / "documentation"
    docs_dir.mkdir(exist_ok=True)
    for doc in ["xray_full_project_proposal.md", "xray_technical_guides.md", "xray_client_briefings.md"]:
        if (root / "docs" / doc).exists():
            shutil.copy(root / "docs" / doc, docs_dir / doc)
    print("✅ Successfully gathered Strategic Proposals.")

    print("\n" + "="*60)
    print("🚀 DONE: Delivery package is ready at: Milestone4_Delivery_Package/")
    print("Action: Download this folder to your local PC for the final hand-off.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
