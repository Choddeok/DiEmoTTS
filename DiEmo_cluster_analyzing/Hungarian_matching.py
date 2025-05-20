import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

# Read the data file (assuming the txt file is in CSV format)
df = pd.read_csv("./VAD_list/Final_relative_positional_information.txt")

speaker_pairs = [(11, 12), (12, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 20)]

# Open output text file for writing results
output_filepath = "./results/emotion_matching_results.txt"
with open(output_filepath, "w", encoding="utf-8") as fout:
    for speaker1, speaker2 in speaker_pairs:
        df1 = df[df['speaker'] == speaker1].reset_index(drop=True)
        df2 = df[df['speaker'] == speaker2].reset_index(drop=True)
    
        # Check if both speakers have data
        if df1.empty or df2.empty:
            fout.write(f"No data for speaker {speaker1} or speaker {speaker2}.\n\n")
            continue
    
        # Assume that the number of emotions for both speakers is the same
        n = len(df1)
        cost_matrix = np.zeros((n, n))
    
        # Calculate the Euclidean distance between the azimuth and elevation angles for each emotion
        for i in range(n):
            for j in range(n):
                angle_diff = df1.loc[i, 'azimuth'] - df2.loc[j, 'azimuth']
                elevation_diff = df1.loc[i, 'elevation'] - df2.loc[j, 'elevation']
                cost_matrix[i, j] = np.sqrt(angle_diff**2 + elevation_diff**2)
    
        # Apply the Hungarian algorithm (minimum cost matching)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
        # Write header for the current speaker pair
        print(f"Emotion matching results between speaker {speaker1} and speaker {speaker2}:\n")
        fout.write(f"Emotion matching results between speaker {speaker1} and speaker {speaker2}:\n")
        for i, j in zip(row_ind, col_ind):
            emotion1 = df1.loc[i, 'predicted_group']
            emotion2 = df2.loc[j, 'predicted_group']
            cost = cost_matrix[i, j]
            print(f"  {emotion1} (speaker {speaker1})  ->  {emotion2} (speaker {speaker2})  [cost: {cost:.2f}]\n")
            fout.write(f"  {emotion1} (speaker {speaker1})  ->  {emotion2} (speaker {speaker2})  [cost: {cost:.2f}]\n")
        fout.write("\n")  # Blank line between speaker pairs

print(f"Final matching results have been written to {output_filepath}")
