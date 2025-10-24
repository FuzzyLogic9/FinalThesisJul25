import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Simulated content loaders (since we can't access user's local paths here)
def parse_matrix_file_simulated(file_path):
    # Use uploaded structure: Matrix ID header + 4x4 rows
    with open(file_path, 'r') as f:
        lines = f.readlines()

    matrices = []
    matrix = []
    matrix_id = None

    for line in lines:
        line = line.strip()
        if line.startswith("Matrix ID:"):
            if matrix and matrix_id is not None:
                matrices.append((matrix_id, np.array(matrix, dtype=float)))
                matrix = []
            matrix_id = int(line.split(":")[1].strip())
        elif line:
            matrix.append(list(map(float, line.strip().split())))
    # Append last one
    if matrix and matrix_id is not None:
        matrices.append((matrix_id, np.array(matrix, dtype=float)))

    return matrices

def load_bitstring_simulated(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()

def normalize_matrix(mat):
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid divide-by-zero
    return mat / row_sums

def compute_ksev_from_matrices(matrices, bitstring):
    group0 = []
    group1 = []

    for bit, (mid, mat) in zip(bitstring, matrices):
        if bit == '0':
            group0.append(mat)
        elif bit == '1':
            group1.append(mat)

    global0 = sum(group0)
    global1 = sum(group1)

    norm0 = normalize_matrix(global0)
    norm1 = normalize_matrix(global1)

    p_values = []
    for i in range(4):
        _, p = ks_2samp(norm0[i], norm1[i], mode='asymp')
        p_values.append(p)

    ksev_score = np.mean(p_values)
    return ksev_score, p_values, norm0, norm1

# Paths for simulation (adapt for real ones when deploying)
#matrix_file_path = '/mnt/data/Sample1661_transposed_matrices_by_4.txt'
#bitstring_file_path = '/mnt/data/binary5000.txt'
matrix_file_path = 'C:/Users/fzkar/final_code_data/Sample1661_transposed_matrices_by_4.txt'
bitstring_file_path = 'C:/Users/fzkar/final_code_data/binary5000.txt'

# Simulate function call (actual files must exist for local run)
matrices = parse_matrix_file_simulated(matrix_file_path)
bitstring = load_bitstring_simulated(bitstring_file_path)
# === Step 2: Compute KSEV ===
ksev_score, p_vals, norm0, norm1 = compute_ksev_from_matrices(matrices, bitstring)

# === Step 3: Output ===
print(f"\n✅ Final KSEV Score (mean p-value, lower = fitter): {ksev_score:.6f}")
for i, p in enumerate(p_vals):
    print(f"Row {i} KS p-value: {p:.6f}")
# For now, just show function structure ready to run
"✅ KSEV Python function structure built and ready for file testing."
