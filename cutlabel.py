import csv
import os
import json

config_path = "pipeline_config.json"
sample_id = "mouse_brain_demo"
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sample_id = cfg.get("SAMPLE_ID", "mouse_brain_demo")

input_file = f"bin_edge_labels_{sample_id}.csv"
label_stats_path = f"label_stats_{sample_id}.json"

total_pos_samples = 0
cut_ratio = 20  # Default value

if os.path.exists(label_stats_path):
    try:
        with open(label_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        total_pos_samples = stats.get("pos_samples", 0)
        total_sub_samples = stats.get("sub_samples", 0)

        # Dynamically calculate cut_ratio: assuming the target is 1/N of positive samples,
        # e.g., if N=10, we keep 1/10 of the positive samples.
        # If total_pos_samples is very small, for example less than 10, then do not split.
        if total_pos_samples > 10:
            cut_ratio = int(total_pos_samples / total_sub_samples)  # Target is to keep 1/10 of total positive samples
            if cut_ratio == 0: cut_ratio = 1  # Keep at least one
        else:
            cut_ratio = 1  # Too few samples, retain all positive samples directly

        # Update output filename to reflect the splitting ratio
        output_file = f"bin_edge_labels_mouse_demo_cut12.csv"

    except Exception as e:
        print(f"[WARN] Failed to read label_stats.json, default splitting ratio of 20 will be used: {e}")
cut_ratio = cut_ratio / 2
output_file = f"bin_edge_labels_{sample_id}_cut12.csv"
print(f"[INFO] Total positive samples in current dataset: {total_pos_samples}, retaining approximately 1/{cut_ratio} of positive samples.")
# ====================================================================

count_pos = 0

with open(input_file, "r", newline='', encoding="utf-8") as fin, \
        open(output_file, "w", newline='', encoding="utf-8") as fout:
    # ... (Remaining code is consistent with before, using dynamically calculated cut_ratio) ...
    reader = csv.reader(fin, delimiter=",")
    writer = csv.writer(fout, delimiter=",")

    header = next(reader, None)
    if header:
        writer.writerow(header)

    for row in reader:
        if not row:
            continue

        label_str = row[-1].strip()

        try:
            label = int(label_str)
        except ValueError:
            continue

        if label == -1:
            writer.writerow(row)

        elif label == 1:
            if count_pos % cut_ratio == 0:  # Use dynamic cut_ratio
                writer.writerow(row)
            count_pos += 1
