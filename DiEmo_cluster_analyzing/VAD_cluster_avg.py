import csv
import os
from collections import defaultdict

# Folder path where the files are located (modify as needed)
folder = './VAD_list'

# Process files from 0011 to 0020
for speaker in range(11, 21):
    speaker_str = f"{speaker:04d}"  # e.g., "0011", "0012", ..., "0020"
    input_filename = os.path.join(folder, f"VAD_cluster_{speaker_str}.txt")
    
    if not os.path.exists(input_filename):
        print(f"{input_filename} file does not exist.")
        continue

    # Dictionary to store A_trans, V_trans, D_trans values for each group 
    # (predicted_cluster (predicted_emotion))
    group_values = defaultdict(list)
    
    with open(input_filename, 'r', encoding='utf-8') as fin:
        reader = csv.reader(fin)
        header = next(reader, None)  # Skip header if present
        for row in reader:
            # Each row is in the format: [speaker_label, predicted_group, A_trans, V_trans, D_trans]
            group = row[2]  # f"{info['predicted_cluster']} ({info['predicted_emotion']})"
            try:
                A_trans = float(row[3])
                V_trans = float(row[4])
                D_trans = float(row[5])
            except ValueError:
                continue
            group_values[group].append((A_trans, V_trans, D_trans))
    
    # Calculate averages for each group
    output_filename = os.path.join(folder, f"VAD_cluster_avg_{speaker_str}.txt")
    with open(output_filename, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout)
        # Write header
        writer.writerow(["predicted_group", "avg_A", "avg_V", "avg_D"])
        for group, values in group_values.items():
            count = len(values)
            avg_A = sum(v[0] for v in values) / count if count > 0 else 0
            avg_V = sum(v[1] for v in values) / count if count > 0 else 0
            avg_D = sum(v[2] for v in values) / count if count > 0 else 0
            writer.writerow([group, avg_A, avg_V, avg_D])
    
    print(f"{input_filename} processed - average results saved in: {output_filename}")
