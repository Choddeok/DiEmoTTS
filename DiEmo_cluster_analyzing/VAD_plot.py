import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

folder = './VAD_list'
plot = './plot'

for speaker in range(11, 21):
    speaker_str = f"{speaker:04d}"
    filename = os.path.join(folder, f"VAD_cluster_avg_{speaker_str}.txt")
    
    if not os.path.exists(filename):
        print(f"{filename} file does not exist.")
        continue

    groups = []
    A_coords = []
    V_coords = []
    D_coords = []
    with open(filename, 'r', encoding='utf-8') as f_in:
        reader = csv.reader(f_in)
        header = next(reader, None)  # Skip header if present
        for row in reader:
            # Check data format
            groups.append(row[0])
            try:
                A_coords.append(float(row[1]))
                V_coords.append(float(row[2]))
                D_coords.append(float(row[3]))
            except ValueError:
                continue

    if not A_coords:
        print(f"No data available for speaker {speaker_str}.")
        continue

    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(A_coords, V_coords, D_coords, c='blue', marker='o')
    ax.set_title(f"Aggregated Coordinates for Speaker {speaker_str}")
    ax.set_xlabel('avg_A')
    ax.set_ylabel('avg_V')
    ax.set_zlabel('avg_D')

    # Optionally, display predicted group information above each point
    for i, group in enumerate(groups):
        ax.text(A_coords[i], V_coords[i], D_coords[i], group, size=9, zorder=1, color='red')

    # Save plot to file instead of using plt.show()
    output_plot_filename = os.path.join(plot, f"aggregated_plot_{speaker_str}.png")
    plt.savefig(output_plot_filename)
    plt.close()  # Close figure to free up memory
    print(f"Speaker {speaker_str} plot saved as {output_plot_filename}")
