import csv
import os
from scipy.spatial import ConvexHull

# Folder where aggregated output files are located (e.g., ./list)
folder = './VAD_list'
result = './results'

# Process speakers from 0011 to 0020
for speaker in range(11, 21):
    speaker_str = f"{speaker:04d}"  # e.g., "0011", "0012", ..., "0020"
    input_filename = os.path.join(folder, f"VAD_cluster_avg_{speaker_str}.txt")
    
    if not os.path.exists(input_filename):
        print(f"{input_filename} file does not exist.")
        continue

    # Read file: each row is [speaker_label, predicted cluster (predicted Emotion), avg_A, avg_V, avg_D]
    cluster_points = []  # Each element: (predicted_group, [avg_A, avg_V, avg_D])
    with open(input_filename, 'r', encoding='utf-8') as fin:
        reader = csv.reader(fin)
        for row in reader:
            predicted_group = row[0]
            try:
                avg_A = float(row[1])
                avg_V = float(row[2])
                avg_D = float(row[3])
            except ValueError:
                continue
            cluster_points.append((predicted_group, [avg_A, avg_V, avg_D]))
    
    if not cluster_points:
        print(f"No data available for speaker {speaker_str}.")
        continue

    # Dictionary to store convex hull areas when excluding each cluster
    candidate_results = {}  # Key: candidate cluster, Value: convex hull area

    # Calculate convex hull area of remaining points after excluding each cluster one by one
    for candidate, _ in cluster_points:
        # Gather points from clusters excluding the candidate cluster
        points_remaining = [pt for grp, pt in cluster_points if grp != candidate]
        if len(points_remaining) < 4:
            area = 0  # A 3D convex hull requires at least 4 points
        else:
            try:
                hull = ConvexHull(points_remaining)
                area = hull.area  # Surface area of the convex hull
            except Exception as e:
                area = 0
        candidate_results[candidate] = area

    # Identify the candidate cluster that yields the largest convex hull area
    best_candidate = max(candidate_results, key=lambda k: candidate_results[k])
    best_area = candidate_results[best_candidate]

    # Save results to file (e.g., convex_hull_result_0011.txt)
    output_filename = os.path.join(result, f"convex_hull_result_{speaker_str}.txt")
    with open(output_filename, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout)
        # Header: speaker_label, candidate_excluded_cluster, convex_hull_area, best_candidate_flag
        writer.writerow(["speaker_label", "candidate_excluded_cluster", "convex_hull_area", "best_candidate_flag"])
        for candidate, area in candidate_results.items():
            best_flag = "BEST" if candidate == best_candidate else ""
            writer.writerow([speaker_str, candidate, area, best_flag])
    
    print(f"Processing complete for speaker {speaker_str}: convex hull areas by exclusion - {candidate_results}, BEST: {best_candidate} (area: {best_area})")
