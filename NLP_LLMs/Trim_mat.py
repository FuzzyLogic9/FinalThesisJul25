import numpy as np
import pandas as pd

# File paths
input_path = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/GPT2_G1.txt"  # Replace with your actual file
output_dir_full = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/GPT2_G1.csv"
output_dir_trimmed = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/4GPT2_G1.csv"

# Initialize output lists
full_output = []
trimmed_output = []

# Indices to keep for 4x4
keep_indices = [0, 3, 4, 7]

with open(input_path, 'r') as file:
    lines = file.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.isdigit():  # It's a matrix ID
        matrix_id = int(line)
        i += 1
        matrix_data = []

        # Read the next 8 lines
        for _ in range(8):
            row = list(map(int, lines[i].strip().split()))
            matrix_data.append(row)
            i += 1

        matrix_array = np.array(matrix_data)

        # Store full 8x8 with ID
        full_df = pd.DataFrame(matrix_array)
        full_df.insert(0, "MatrixID", matrix_id)
        full_output.append(full_df)

        # Trim to 4x4
        reduced_array = matrix_array[np.ix_(keep_indices, keep_indices)]
        trimmed_df = pd.DataFrame(reduced_array)
        trimmed_df.insert(0, "MatrixID", matrix_id)
        trimmed_output.append(trimmed_df)

    else:
        i += 1  # Skip malformed lines or empty lines

# Concatenate and save all matrices
pd.concat(full_output).to_csv(output_dir_full, index=False)
pd.concat(trimmed_output).to_csv(output_dir_trimmed, index=False)

print("Done. Files saved:")
print(f" - Full matrices: {output_dir_full}")
print(f" - Trimmed matrices: {output_dir_trimmed}")
