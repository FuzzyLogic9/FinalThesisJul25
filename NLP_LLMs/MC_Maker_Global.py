import pandas as pd
import numpy as np

# File paths
input_file = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/Page_Paired_GPTNEO_FT_G1_OT.csv"
output_path = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/GPTNEO_FT_G1_OT_GLOBAL_FULL.txt"
fixed_output_path = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/GPTNEO_FT_G1_OT_FIXED_4x4.txt"
dynamic_output_path = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/GPTNEO_FT_G1_OT_DYNAMIC_4x4.txt"

# Read the CSV file
df = pd.read_csv(input_file)
print("First few rows of the CSV file:")
print(df.head())
print("\nColumns in CSV file:", df.columns)

# Initialize the global 8x8 matrix
global_matrix = np.zeros((8, 8), dtype=int)

# Populate the matrix
for index, row in df.iterrows():
    pair = row['Pairs for MC']
    
    if isinstance(pair, str) and ',' in pair:
        coords = pair.split(',')
        if len(coords) == 2:
            try:
                x, y = int(coords[0]), int(coords[1])
                if 0 <= x < 8 and 0 <= y < 8:
                    global_matrix[x][y] += 1
            except ValueError:
                print(f"Invalid pair format: {pair}")

# === Save the full matrix ===
with open(output_path, 'w') as f:
    f.write("GLOBAL 8x8 MATRIX\n")
    for row in global_matrix:
        f.write(' '.join(map(str, row)) + '\n')

# === Fixed 4x4 matrix (remove 1,2,5,6) ===
fixed_keep_indices = [0, 3, 4, 7]
fixed_matrix = global_matrix[np.ix_(fixed_keep_indices, fixed_keep_indices)]

with open(fixed_output_path, 'w') as f:
    f.write("FIXED 4x4 MATRIX (States 0,3,4,7)\n")
    for row in fixed_matrix:
        f.write(' '.join(map(str, row)) + '\n')

# === Dynamic 4x4 matrix (remove least active 4 states) ===
row_sums = global_matrix.sum(axis=1)
col_sums = global_matrix.sum(axis=0)
activity = row_sums + col_sums

least_active_indices = np.argsort(activity)[:4]  # 4 lowest
dynamic_keep_indices = [i for i in range(8) if i not in least_active_indices]
dynamic_matrix = global_matrix[np.ix_(dynamic_keep_indices, dynamic_keep_indices)]

with open(dynamic_output_path, 'w') as f:
    f.write(f"DYNAMIC 4x4 MATRIX (Removed states: {least_active_indices.tolist()})\n")
    for row in dynamic_matrix:
        f.write(' '.join(map(str, row)) + '\n')

# === Final Confirmation ===
print(f"\nGlobal 8x8 matrix saved to: {output_path}")
print(f"Fixed 4x4 matrix saved to: {fixed_output_path}")
print(f"Dynamic 4x4 matrix saved to: {dynamic_output_path}")
