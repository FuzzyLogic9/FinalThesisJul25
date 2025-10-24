import numpy as np
import pandas as pd

# File paths
input_path = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/GPT2_G1.txt"  # Change this
output_dir_full = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/GPT2_G1.csv"
output_dir_trimmed = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/4DGPT2_G1.csv"

# Containers for all matrices
full_output = []
trimmed_output = []

with open(input_path, 'r') as file:
    lines = file.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.isdigit():
        matrix_id = int(line)
        i += 1
        matrix_data = []

        for _ in range(8):
            row = list(map(int, lines[i].strip().split()))
            matrix_data.append(row)
            i += 1

        matrix_array = np.array(matrix_data)

        # Save full version
        df_full = pd.DataFrame(matrix_array)
        df_full.insert(0, "MatrixID", matrix_id)
        full_output.append(df_full)

        # Compute state activity: sum of rows + sum of columns
        row_sums = matrix_array.sum(axis=1)
        col_sums = matrix_array.sum(axis=0)
        total_activity = row_sums + col_sums

        # Identify the 4 states with the least activity
        least_active_indices = np.argsort(total_activity)[:4]

        # Remove these states
        keep_indices = [i for i in range(8) if i not in least_active_indices]
        reduced_matrix = matrix_array[np.ix_(keep_indices, keep_indices)]

        # Save reduced version
        df_trimmed = pd.DataFrame(reduced_matrix)
        df_trimmed.insert(0, "MatrixID", matrix_id)
        trimmed_output.append(df_trimmed)

    else:
        i += 1  # Skip empty or malformed lines

# Concatenate and save
pd.concat(full_output).to_csv(output_dir_full, index=False)
pd.concat(trimmed_output).to_csv(output_dir_trimmed, index=False)

print("✔ Done!")
print(f"Saved full matrices to: {output_dir_full}")
print(f"Saved dynamically trimmed matrices to: {output_dir_trimmed}")
