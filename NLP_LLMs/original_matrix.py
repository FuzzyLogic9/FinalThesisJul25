import numpy as np

# === Original normalized matrix with a high self-loop at [0][0]
matrix = np.array([
    [0.92, 0, 0, 0.89, 0.02, 0, 0, 0.09],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0.73, 0, 0, 0.25, 0, 0, 0, 0.02],
    [0.9, 0, 0, 0.06, 0.04, 0, 0, 0.01],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0.7, 0, 0, 0.21, 0.01, 0, 0, 0.09]
])

# === Step 1: Remove the State 0 → State 0 self-loop
matrix[0][0] = 0

# === Step 2: Renormalize row 0
row_sum = matrix[0].sum()
if row_sum > 0:
    matrix[0] = matrix[0] / row_sum

# === Save paths
path_global = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/GLOBAL_MODIFIED.txt"
path_fixed = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/FIXED_MODIFIED_4x4.txt"
path_dynamic = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/DYNAMIC_MODIFIED_4x4.txt"

# === Write GLOBAL matrix
with open(path_global, 'w') as f:
    f.write("GLOBAL MODIFIED 8x8 MATRIX (State 0->0 removed)\n")

    for row in matrix:
        f.write(' '.join(f"{x:.4f}" for x in row) + '\n')

# === Fixed 4x4 matrix: keep only [0, 3, 4, 7]
fixed_indices = [0, 3, 4, 7]
fixed_matrix = matrix[np.ix_(fixed_indices, fixed_indices)]
with open(path_fixed, 'w') as f:
    f.write("FIXED MODIFIED 4x4 MATRIX (States 0,3,4,7)\n")
    for row in fixed_matrix:
        f.write(' '.join(f"{x:.4f}" for x in row) + '\n')

# === Dynamic 4x4 matrix: remove 4 least active states
row_sums = matrix.sum(axis=1)
col_sums = matrix.sum(axis=0)
activity = row_sums + col_sums
least_active = np.argsort(activity)[:4]
dynamic_indices = [i for i in range(8) if i not in least_active]
dynamic_matrix = matrix[np.ix_(dynamic_indices, dynamic_indices)]
with open(path_dynamic, 'w') as f:
    f.write(f"DYNAMIC MODIFIED 4x4 MATRIX (Removed states: {least_active.tolist()})\n")
    for row in dynamic_matrix:
        f.write(' '.join(f"{x:.4f}" for x in row) + '\n')

print("✅ All matrices saved successfully:")
print(f" - GLOBAL:  {path_global}")
print(f" - FIXED:   {path_fixed}")
print(f" - DYNAMIC: {path_dynamic}")
