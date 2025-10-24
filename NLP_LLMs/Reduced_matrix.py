import numpy as np

def reduce_and_save_matrices(input_file, output_file):
    # Read matrices and find zero rows/columns
    matrices = []
    current_matrix = None
    page_ids = []
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_matrix is not None:
                    matrices.append(current_matrix)
                    current_matrix = None
            elif len(line.split()) == 1:
                page_ids.append(int(line))
                current_matrix = []
            else:
                row = [int(x) for x in line.split()]
                current_matrix.append(row)
    
    if current_matrix is not None:
        matrices.append(current_matrix)
    
    # Convert to numpy array
    matrices = [np.array(matrix) for matrix in matrices]
    
    # Indices to remove
    remove_indices = [1,2,5,6] # same as original
    #remove_indices = [3,4,5,6] # same as original
    # Write reduced matrices
    with open(output_file, 'w') as f:
        for idx, page_id in enumerate(page_ids):
            matrix = matrices[idx]
            keep_rows = [i for i in range(matrix.shape[0]) if i not in remove_indices]
            keep_cols = [i for i in range(matrix.shape[1]) if i not in remove_indices]
            
            reduced_matrix = matrix[np.ix_(keep_rows, keep_cols)]
            
            f.write(f"Matrix ID: {page_id}\n")
            for row in reduced_matrix:
                f.write(' '.join(map(str, row)) + '\n')
            f.write('\n')

    print(f"Reduced matrices saved to {output_file}")

# Example usage
# reduce_and_save_matrices('input_file.txt', 'output_file.txt')
base_path = "C:/FuzzyNov2024/toxicity_project/Data/M1G1/"
file1_path = base_path + "Tox_1_1_0_matrices_transposed_WH.txt"
file2_path = base_path + "Tox_1_1_0_reduced_matrices_1_WH.txt"
reduce_and_save_matrices(file1_path, file2_path)