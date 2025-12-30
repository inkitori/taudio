import csv
import re
import argparse
import sys
from typing import Dict, List, Tuple

from evaluate_api_timestamp_any import _metrics_from_prediction

def parse_log_value(line: str) -> float:
    # extracts the 'start' value from the log line using regex. also handles negatives
    match = re.search(r"'start':\s*(-?[0-9\.]+)", line)
    if match:
        return float(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Compute metrics from CSV GT and Log Predictions.")
    parser.add_argument("csv_file", help="Path to the CSV file containing gt_timestamp and duration_seconds")
    parser.add_argument("log_file", help="Path to the Log file containing Model response")
    parser.add_argument("--clamp", action="store_true", help="Clamp prediction between 0 and duration_seconds")
    
    args = parser.parse_args()

    # 1. Load Ground Truth and Duration from CSV
    gt_data: List[Tuple[float, float]] = []
    try:
        with open(args.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    gt = float(row['gt_timestamp'])
                    dur = float(row['duration_seconds'])
                    gt_data.append((gt, dur))
                except ValueError:
                    print("Error parsing CSV Row:")
                    print(row)
    except FileNotFoundError:
        print(f"Error: CSV file '{args.csv_file}' not found.")
        sys.exit(1)

    # 2. Load Predictions from Log File
    predictions: List[float] = []
    try:
        with open(args.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Only process lines that actually contain a model response
                if "Model response:" in line:
                    val = parse_log_value(line)
                    if val is not None:
                        predictions.append(val)
                    else:
                        print(line)
    except FileNotFoundError:
        print(f"Error: Log file '{args.log_file}' not found.")
        sys.exit(1)

    # 3. Validation
    # We zip the lists, so it will only process up to the length of the shortest file
    # assuming the order is preserved 1:1.
    if len(gt_data) != len(predictions):
        print(f"ERROR: Number of CSV entries ({len(gt_data)}) does not match number of Log predictions ({len(predictions)}).")
        sys.exit(1)

    if not gt_data or not predictions:
        print("Error: No valid data found in one or both files.")
        sys.exit(1)

    # 4. Compute Metrics
    aggregated_metrics = {}
    count = 0

    for (gt, duration), pred in zip(gt_data, predictions):
        # Apply Clamping if requested
        if args.clamp:
            pred = max(0.0, min(pred, duration))

        # Get metrics for this single item
        result = _metrics_from_prediction(pred, gt)
        
        # Accumulate
        for k, v in result.items():
            aggregated_metrics[k] = aggregated_metrics.get(k, 0.0) + v
        
        count += 1

    # 5. Average and Output
    print(f"Processed {count} examples.\n")
    print(f"{'Metric':<30} | {'Average':<10}")
    print("-" * 45)

    # Calculate averages
    keys = aggregated_metrics.keys() # don't wanna sort
    
    for key in keys:
        avg_value = aggregated_metrics[key] / count
        
        print(f"{key}: {avg_value}")

if __name__ == "__main__":
    main()