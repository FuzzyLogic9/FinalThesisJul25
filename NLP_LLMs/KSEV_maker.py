import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def load_and_normalize_matrix(file_path):
    mat = pd.read_csv(file_path, header=None).values.astype(float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    return mat / row_sums

def compute_ksev_rowwise_ks(group0_path, group1_path):
    A = load_and_normalize_matrix(group0_path)
    B = load_and_normalize_matrix(group1_path)

    p_values = []
    for i in range(A.shape[0]):
        row_A = A[i]
        row_B = B[i]

        print(f"\n🔍 Row {i} Group 0: {row_A}")
        print(f"🔍 Row {i} Group 1: {row_B}")

        _, p = ks_2samp(row_A, row_B, mode='asymp')  # more stable
        p_values.append(p)

    mean_p = np.mean(p_values)
    return mean_p, p_values

# === Run it ===
g0_file = "C:/Users/fzkar/final_code_data/MC_KSEV_Global_G0_2.csv"
g1_file = "C:/Users/fzkar/final_code_data/MC_KSEV_Global_G1_2.csv"

ksev_score, rowwise_p = compute_ksev_rowwise_ks(g0_file, g1_file)

print("\n✅ Row-wise KSEV p-values:", [round(p, 6) for p in rowwise_p])
print(f"✅ Final KSEV Score (mean p-value, lower = fitter): {ksev_score:.6f}")
