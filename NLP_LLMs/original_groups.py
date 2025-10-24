import numpy as np

def load_matrices_from_file(filepath):
    matrices = []
    current_matrix = []
    current_id = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Matrix ID:"):
                if current_id is not None and current_matrix:
                    matrices.append((current_id, np.array(current_matrix, dtype=int)))
                    current_matrix = []
                current_id = int(line.split(":")[1].strip())
            else:
                current_matrix.append(list(map(int, line.split())))
        
        # Add the last matrix
        if current_id is not None and current_matrix:
            matrices.append((current_id, np.array(current_matrix, dtype=int)))

    return matrices

def split_and_sum_by_bitstring(matrices, bitstring):
    group_0_matrices = []
    group_1_matrices = []
    group_0_ids = []
    group_1_ids = []

    for bit, (mid, mat) in zip(bitstring, matrices):
        if bit == '0':
            group_0_matrices.append(mat)
            group_0_ids.append(mid)
        elif bit == '1':
            group_1_matrices.append(mat)
            group_1_ids.append(mid)

    global_0 = sum(group_0_matrices) if group_0_matrices else np.zeros((4, 4), dtype=int)
    global_1 = sum(group_1_matrices) if group_1_matrices else np.zeros((4, 4), dtype=int)

    return group_0_ids, global_0, group_1_ids, global_1

# === USAGE ===
matrix_file = 'C:/Users/fzkar/final_code_data/Sample1661_transposed_matrices_by_4.txt'
bitstring_file = 'C:/Users/fzkar/final_code_data/binary5000.txt'

# Load matrix data
matrices = load_matrices_from_file(matrix_file)

# Load the 1662-bit string
with open(bitstring_file, 'r') as f:
    bitstring = f.read().strip()

# Sanity check
assert len(matrices) == len(bitstring) == 1662, "Mismatch in number of matrices and bits"

# Process
group_0_ids, global_0, group_1_ids, global_1 = split_and_sum_by_bitstring(matrices, bitstring)

# Print summary
print("Group 0 (Label 0) Matrix IDs:", group_0_ids[:5], "... total:", len(group_0_ids))
print("Group 1 (Label 1) Matrix IDs:", group_1_ids[:5], "... total:", len(group_1_ids))
print("Global Matrix for Group 0:\n", global_0)
print("Global Matrix for Group 1:\n", global_1)
import pandas as pd

# Save global matrices to CSV like MC_Global format
def save_global_matrix_csv(matrix, output_file):
    df = pd.DataFrame(matrix.astype(int))
    df.to_csv(output_file, index=False, header=False)

# Paths for output
global0_csv = 'C:/Users/fzkar/final_code_data/MC_KSEV_Global_G0_2.csv'
global1_csv = 'C:/Users/fzkar/final_code_data/MC_KSEV_Global_G1_2.csv'

# Save them
save_global_matrix_csv(global_0, global0_csv)
save_global_matrix_csv(global_1, global1_csv)

print("✅ Global matrices saved as MC_Global_G0.csv and MC_Global_G1.csv")
