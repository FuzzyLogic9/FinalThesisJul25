import numpy as np

def transpose_matrices(input_file, output_file):
    matrices = []
    current_matrix = None
    page_ids = []
    
    # Read matrices
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
    
    # Write transposed matrices
    with open(output_file, 'w') as f:
        for idx, page_id in enumerate(page_ids):
            matrix = np.array(matrices[idx])
            transposed_matrix = matrix.T
            
            f.write(f"{page_id}\n")
            for row in transposed_matrix:
                f.write(' '.join(map(str, row)) + '\n')
            f.write('\n')

input_file = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/GPT2_G1.txt"
output_file = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/X_GPT2_G1.txt"

transpose_matrices(input_file, output_file)
print("Transposed matrices saved")