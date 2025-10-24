import pandas as pd
import numpy as np

input_file = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/Page_Paired_GPT2_G1_OT.csv"
output_file = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/GPT2_G1.txt"

# Read Excel file
df = pd.read_csv(input_file)
print("First few rows of the CSV file:")
print(df.head())
print("\nColumns in CSV file:", df.columns)

# Create matrices
matrices = {}
for index, row in df.iterrows():
    page_id = row['PageID']
    pair = row['Pairs for MC']
    
    if page_id not in matrices:
        matrices[page_id] = np.zeros((8, 8), dtype=int)
    
    if isinstance(pair, str) and ',' in pair:
        coords = pair.split(',')
        if len(coords) == 2:
            try:
                x, y = int(coords[0]), int(coords[1])
                if 0 <= x < 8 and 0 <= y < 8:
                    matrices[page_id][x][y] += 1
            except ValueError:
                print(f"Invalid pair format: {pair}")

# Write matrices to file
with open(output_file, 'w') as f:
    for page_id, matrix in matrices.items():
        f.write(f"{page_id}\n")
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')
        f.write('\n')

print(f"\nProcessed {len(matrices)} unique PageIDs")