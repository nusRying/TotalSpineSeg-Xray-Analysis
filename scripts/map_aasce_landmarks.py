import csv
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Map AASCE coordinates to canonical CSV schema.")
    parser.add_argument("--landmarks-csv", type=Path, required=True, help="AASCE landmarks.csv file.")
    parser.add_argument("--filenames-csv", type=Path, required=True, help="AASCE filenames.csv file.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output canonical CSV file.")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load filenames
    with open(args.filenames_csv, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f if line.strip()]

    # 2. Process landmarks
    fieldnames = ["case_id", "vertebra_order", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    
    with open(args.output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with open(args.landmarks_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if row_idx >= len(filenames):
                    break
                
                case_id = Path(filenames[row_idx]).stem
                # AASCE: 68 X values, then 68 Y values
                coords = [float(x) for x in row]
                xs = coords[:68]
                ys = coords[68:]

                # 17 vertebrae, 4 points each
                for v_idx in range(17):
                    start = v_idx * 4
                    # Extract 4 corners: TL, TR, BR, BL
                    v_xs = xs[start:start+4]
                    v_ys = ys[start:start+4]
                    
                    writer.writerow({
                        "case_id": case_id,
                        "vertebra_order": v_idx + 1,
                        "x1": v_xs[0], "y1": v_ys[0],
                        "x2": v_xs[1], "y2": v_ys[1],
                        "x3": v_xs[2], "y3": v_ys[2],
                        "x4": v_xs[3], "y4": v_ys[3]
                    })

    print(f"✅ Mapped {len(filenames)} AASCE cases to {args.output_csv}")

if __name__ == "__main__":
    main()
