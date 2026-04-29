import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import shutil

def generate_plots(metrics_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(metrics_csv)
    
    # 1. Dice Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['dice'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Dice Scores (Vertebrae Segmentation)', fontsize=15)
    plt.xlabel('Dice Score', fontsize=12)
    plt.ylabel('Number of Cases', fontsize=12)
    plt.axvline(df['dice'].mean(), color='red', linestyle='--', label=f'Mean: {df["dice"].mean():.3f}')
    plt.legend()
    plt.savefig(output_dir / 'dice_distribution.png')
    plt.close()
    
    # 2. Precision vs Recall
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x='recall', y='precision', alpha=0.5, color='darkblue')
    plt.title('Precision vs. Recall (Vertebrae Detection)', fontsize=15)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_dir / 'precision_recall_scatter.png')
    plt.close()
    
    # 3. Best/Worst Case Identification
    df_sorted = df.sort_values(by='dice', ascending=False)
    best_cases = df_sorted.head(5)['case_id'].tolist()
    worst_cases = df_sorted.tail(5)['case_id'].tolist()
    
    print(f"Top 5 Cases: {best_cases}")
    print(f"Bottom 5 Cases: {worst_cases}")
    
    return best_cases, worst_cases

def create_gallery(case_ids, src_dir, dst_dir, label="best"):
    dst_dir = Path(dst_dir) / label
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(src_dir)
    
    for case_id in case_ids:
        # Looking for preview images
        preview_path = src_dir / f"{case_id}.png"
        if preview_path.exists():
            shutil.copy(preview_path, dst_dir / f"{case_id}_preview.png")

if __name__ == "__main__":
    metrics_path = "data/xray_inference/milestone3_final_report/per_case_metrics.csv"
    output_plots = "data/xray_inference/milestone3_final_report/plots"
    preview_dir = "data/xray_inference/milestone3_final_test/preview"
    
    best, worst = generate_plots(metrics_path, output_plots)
    create_gallery(best, preview_dir, output_plots, "best_cases")
    create_gallery(worst, preview_dir, output_plots, "worst_cases")
