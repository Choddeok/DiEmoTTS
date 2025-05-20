import os
import csv
import math

# Folder where the files are located (modify if necessary)
folder = "./VAD_list"

# Process files from 0011 to 0020
for num in range(11, 21):
    filename = os.path.join(folder, f"VAD_cluster_avg_{num:04d}.txt")
    if not os.path.exists(filename):
        print(f"{filename} file does not exist.")
        continue

    data = []
    neutral = None  # Reference Neutral coordinate (line with predicted_group containing "Neutral")

    # Read file: each row is [predicted_group, avg_A, avg_V, avg_D]
    with open(filename, 'r', encoding='utf-8') as fin:
        reader = csv.reader(fin)
        for row in reader:
            if len(row) < 4:
                continue
            predicted_group = row[0].strip()
            try:
                avg_A = float(row[1])
                avg_V = float(row[2])
                avg_D = float(row[3])
            except ValueError:
                continue

            data.append((predicted_group, avg_A, avg_V, avg_D))
            # Find reference group: if "Neutral" is in predicted_group, set it as the reference coordinate
            if "Neutral" in predicted_group:
                neutral = (avg_A, avg_V, avg_D)

    if neutral is None:
        print(f"Could not find reference Neutral (group 3) in {filename}.")
        continue

    output_filename = os.path.join(folder, f"VAD_cluster_avg_{num:04d}_spherical.txt")
    with open(output_filename, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout)
        # Header: predicted_group, r, azimuth, elevation
        writer.writerow(["speaker", "predicted_group", "azimuth", "elevation"])

        # For each row, subtract the reference coordinate and convert to spherical coordinates
        for predicted_group, avg_A, avg_V, avg_D in data:
            # Subtract the reference (neutral) coordinate to perform the translation
            x = avg_A - neutral[0]
            y = avg_V - neutral[1]
            z = avg_D - neutral[2]

            # Compute the radius
            r = math.sqrt(x**2 + y**2 + z**2)
            # Compute the azimuth: the angle in the x-y plane (using atan2, in degrees)
            azimuth = math.degrees(math.atan2(y, x)) if not (x == 0 and y == 0) else 0
            # Compute the elevation: the angle between the vector and the horizontal plane (x-y plane)
            horizontal = math.sqrt(x**2 + y**2)
            elevation = math.degrees(math.atan2(z, horizontal)) if horizontal != 0 else (90 if z > 0 else -90)

            writer.writerow([num, predicted_group, azimuth, elevation])

    print(f"{filename} -> {output_filename} conversion complete")


# --- Step 2: Merge all generated spherical files into one, excluding rows with 'Neutral' ---
merged_filename = os.path.join(folder, "Final_relative_positional_information.txt")

with open(merged_filename, 'w', encoding='utf-8', newline='') as fout:
    writer = csv.writer(fout)
    # Write header: speaker, predicted_group, r, azimuth, elevation
    writer.writerow(["speaker", "predicted_group", "azimuth", "elevation"])

    # Process spherical files from 0011 to 0020
    for num in range(11, 21):
        spherical_filename = os.path.join(folder, f"VAD_cluster_avg_{num:04d}_spherical.txt")
        if not os.path.exists(spherical_filename):
            print(f"{spherical_filename} file does not exist.")
            continue

        with open(spherical_filename, 'r', encoding='utf-8') as fin:
            reader = csv.reader(fin)
            header = next(reader, None)  # Skip header if present
            for row in reader:
                # row is expected as [speaker, predicted_group, r, azimuth, elevation]
                # Exclude rows where predicted_group contains "Neutral"
                if "Neutral" in row[1]:
                    continue
                writer.writerow(row)

print(f"Merged file created: {merged_filename}")