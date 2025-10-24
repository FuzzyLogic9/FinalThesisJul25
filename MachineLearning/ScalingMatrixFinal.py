import csv
import numpy as np
import pandas as pd
import json
import os

import time
time.sleep(5)  

print("project printed immediately.")
os.system("pause")
print("project.")
  
dfread = pd.read_csv('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661.csv',dtype={'X': 'Int32', 'Y': 'Int32'})  

print(dfread.head)
docnolist= dfread['D'].values.tolist()
print(docnolist)

unique_doc_list = set(docnolist)
print(unique_doc_list)
unique_doc_count = len(unique_doc_list)
print ('No of docs = ')
print (unique_doc_count)
#os.system("pause")

print("Paused")
# time.sleep(10)  
i = input("Press Enter to continue: ")

# Read the data from the CSV file
data = []
with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        data.append([row[0], int(row[1]), int(row[2])])

# Organize the data by the keys
result = {}
for item in data:
    key = item[0]
    values = item[1:]
    if key not in result:
        result[key] = []
    result[key].append(values)

# Transform each list of values into a coordinate matrix and store in a dictionary
matrices = {}
for key, values in result.items():
    matrix = np.zeros((8, 8), dtype=int)
    
    for value_pair in values:
        row, col = value_pair
        matrix[row][col] += 1

    matrices[key] = matrix.tolist()
print (matrix[0])
print("Paused")
#time.sleep(10)  
i = input("Press Enter to continue: ")

# Write matrices to a JSON file
with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_matrices.json', 'w') as file:
    json.dump(matrices, file, indent=4)


# Open the JSON file for reading
with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_matrices.json', 'r') as file:
    # Load the data from the file
    matrices_from_file = json.load(file)
    transposed_matrices_dict = {}  # Initialize an empty dictionary to store the transposed matrices with their keys

with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_transposed_matrices.txt', 'w') as output_file:
    # Access the data using the keys
    for key, matrix in matrices_from_file.items():
        output_file.write(key + '\n')
        transposed_matrix = np.array(matrix).T  # Transpose the matrix using numpy
        transposed_matrices_dict[key] = transposed_matrix.tolist()  # Store the transposed matrix in the dictionary
        for row in transposed_matrix:
            output_file.write(' '.join(map(str, row.tolist())) + '\n')  # Convert numpy array row to string and write
        output_file.write('\n')

all_transposed_matrices = [transposed_matrices_dict]
with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_transposed_matrices_dict.txt', 'w') as output_file2:
     print(all_transposed_matrices,file=output_file2)  # To verify the content

with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_transposed_matrices.json', 'w') as file:
    json.dump(all_transposed_matrices, file, indent=4)

# Adding permutations of the matrices

import itertools

# Load the matrices from your JSON file
with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_transposed_matrices.json', 'r') as file:
    transposed_matrices_tot = json.load(file)

# Generate the permutations of a binary string of length 4
##binary_permutations = list(itertools.product([0, 1], repeat=4))
import random

def generate_unique_binary_strings(num_strings, length):
    # Set to keep track of unique binary strings
    unique_binary_strings = set()

    # Keep generating strings until we have num_strings unique ones
    while len(unique_binary_strings) < num_strings:
        # Generate a random binary string of the specified length
        binary_string = ''.join(random.choice('01') for _ in range(length))
        # Add the generated string to the set if it's not already present
        unique_binary_strings.add(binary_string)

    # Save and Return the set as a list
    with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_binarylist_dict.txt', 'w') as output_file2:
        print(list(unique_binary_strings),file=output_file2)

    return list(unique_binary_strings)

# Usage
#num_strings = 1000
num_strings = 1
length = 1662
#unique_binary_strings = generate_unique_binary_strings(num_strings, length)
unique_binary_strings= "010100101110001110110101110110001011011101110001111100000000100011101111110011100100010101001010100001001001011100101010010000110111001000100111101100001110100100000011110010111000100000111101111111001010001001001000110010100100010111001001111110000100100101000100000111001100000001001100100110000001100011001111001000010000001000100000000110001100110010100101111110000000000111101101110100011011111100101110111011010101000100110100010000101011000011000111010001001011100001100100001001011010000001001101001000101001000000011001010001011101001001110001101010001110010110010010010000011111011100111100111110101011010110100011001111100000101000111101001100110110101000111011101011111101110101000111001000010100001000100110110010010011110111100100000011101000011100000101100011100111011110111000000110101101001000000010000100100001110111011110101010001000000111011100111100111111110000000001010100101010010111011000111111110101110110001001101111001100011010110110000001110110010110011000011010001101111011110101011111111111101111001010101101010111101110110001111101001000011010101011101100010000101011100001111001000001100101010100010011101111010010100111010111000010100001010010001100100001001110000010100001101001000101000010110010111001110011000101110011100010111100101111011001000100111100111011111110011001011001100001111100000111011100011010010110011011001001111011011011001000111101011111111011011000110100111111011011011001011001011111100111110011000100101111000000101000100101110000111110010010100111011010010011101011000011000100100110101010001000100011010011010101000001000101001001100100010100111000010011110010101010011000001101110100100001001101100110"
# The unique_binary_strings list now contains 1000 unique binary strings of length 1662

grouped_results = {}

total_sum_group0 = np.zeros_like(np.array(list(transposed_matrices_dict.values())[0]))  # Initialize with zeros
total_sum_group1 = np.zeros_like(np.array(list(transposed_matrices_dict.values())[0]))  # Initialize with zeros

#for perm in binary_permutations:
for perm in unique_binary_strings:  
    group_0_sum = np.zeros_like(np.array(list(transposed_matrices_dict.values())[0]))  # Initialize with zeros
    group_1_sum = np.zeros_like(np.array(list(transposed_matrices_dict.values())[0]))  # Initialize with zeros
    
    for idx, value in enumerate(perm):
        matrix_key = list(transposed_matrices_dict.keys())[idx]  # Get the key name based on index
        #print ('value: ', value)
        #print("Paused") 
        #time.sleep(10)  
        #i = input("Press Enter to continue: ")
        if value == '0':
            group_0_sum += np.array(transposed_matrices_dict[matrix_key])
            total_sum_group0 += np.array(transposed_matrices_dict[matrix_key])
     
        else:
            group_1_sum += np.array(transposed_matrices_dict[matrix_key])
            total_sum_group1 += np.array(transposed_matrices_dict[matrix_key])
    
    
    # Transpose the summed matrices
    group_0_sum = np.transpose(group_0_sum)
    group_1_sum = np.transpose(group_1_sum)
    
    grouped_results[str(perm)] = {
        "Group0": group_0_sum.tolist(),
        "Group1": group_1_sum.tolist()
    }
    print(f"Permutation: {perm}")
    print("Group0 Matrix:")
    for row in group_0_sum:
        print(row)
    print("\nGroup1 Matrix:")
    for row in group_1_sum:
        print(row)
    print("-------------------------")

# Print total sums for Group0 and Group1
print("Total Sum for Group0:")
for row in total_sum_group0:
    print(row)
print("\nTotal Sum for Group1:")
for row in total_sum_group1:
    print(row)

# Save the transposed, grouped, and total summed results to a JSON file
results_with_totals = {
    "GroupedResults": grouped_results,
    "TotalSumGroup0": total_sum_group0.tolist(),
    "TotalSumGroup1": total_sum_group1.tolist()
}


# Save the transposed, grouped results to a JSON file
with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_grouped_transposed_matrices_results.json', 'w') as file:
    json.dump(grouped_results, file, indent=4)


# To view the results:
for key, value in grouped_results.items():
    print(f"For Binary Permutation: {key}")
    print("Group 0 Matrix:")
    for row in value["Group0"]:
        print(row)
    print("Group 1 Matrix:")
    for row in value["Group1"]:
        print(row)
    print("\n")

# Save the results to a txt file
with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_grouped_matrices_results.txt', 'w') as file:
    for key, value in grouped_results.items():
        file.write(f"For Binary Permutation: {key}\n")
        file.write("Group 0 Matrix:\n")
        for row in value["Group0"]:
            file.write(str(row) + "\n")
        file.write("Group 1 Matrix:\n")
        for row in value["Group1"]:
            file.write(str(row) + "\n")
        file.write("\n")

#with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_transposed_matrices_dict.txt', 'w') as output_file2:
#     print(all_transposed_matrices,file=output_file2)  # To verify the content

#with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_transposed_matrices.json', 'w') as file:
#   data = json.load(file)

# Load the matrices from your JSON file
with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_grouped_transposed_matrices_results.json', 'r') as file:
    print('Sample here')
    data = json.load(file)
    #print(json.dumps(data, indent=4))
# Print only the first key-value pair to avoid overwhelming the console
    first_key = next(iter(data))
    print(json.dumps({first_key: data[first_key]}, indent=4))


print("Paused")
time.sleep(10)  
i = input("Press Enter to continue: ")

# Function to scale matrices by their row sums
def scale_matrices_by_row_sum(data):
    scaled_data = {}
    for perm_key, groups in data.items():
        scaled_groups = {}
        for group_name, matrix in groups.items():
            scaled_matrix = []
            for row in matrix:
                row_sum = sum(row)
                if row_sum == 0:
                    scaled_row = [0] * len(row)  # Or just row, since it's all zeros
                else:
                    scaled_row = [x/row_sum for x in row]
                scaled_matrix.append(scaled_row)
            scaled_groups[group_name] = scaled_matrix
        scaled_data[perm_key] = scaled_groups
    return scaled_data

# Scale all matrices
scaled_matrices = scale_matrices_by_row_sum(data)

# Assume 'scaled_matrices' is your scaled matrices data structured as before

# Convert the scaled data to a string with JSON formatting
scaled_data_string = json.dumps(scaled_matrices, indent=4)

# Write the string to a text file
with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_grouped_transposed_and_scaled_matrices.txt', 'w') as outfile:
    outfile.write(scaled_data_string)

print("Data has been written to Sample1661_grouped_transposed_and_scaled_scaled_matrices.txt")
# Now you can save or print your scaled matrices as a json too
#print(json.dumps(scaled_matrices, indent=4))

# To save the scaled matrices back into a JSON file
with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_scaled_grouped_matrices.json', 'w') as outfile:
    json.dump(scaled_matrices, outfile, indent=4)

#
with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_scaled_grouped_matrices.json', 'r') as file:
   data = json.load(file)

#

def kl_divergence(P, Q):
    # Calculate a safe logarithm which is 0 where Q is 0
    safe_log = np.where(Q > 0, np.log(P / Q), 0)
    # Calculate the KL divergence
    return np.sum(np.where(P > 0, P * safe_log, 0))

# Function to normalize a matrix to a probability distribution
def normalize(matrix):
    s = np.sum(matrix)
    return matrix / s if s > 0 else matrix

# Assume 'data' is your loaded json object containing the groups
# For example:
# data = json.load(open('your_json_file.json'))

# Compute the KL divergence for each pair of matrices
kl_divergences = {}
for key, groups in data.items():
    # Normalize each matrix in both groups to get probability distributions
    group0_normalized = [normalize(np.array(matrix)) for matrix in groups['Group0']]
    group1_normalized = [normalize(np.array(matrix)) for matrix in groups['Group1']]

    # Calculate KL divergence for each corresponding pair in group0 and group1
    kls = [kl_divergence(p, q) for p, q in zip(group0_normalized, group1_normalized)]

    # Store the sum of KL divergences for this key
    kl_divergences[key] = sum(kls)

# Print or save the KL divergences
print(kl_divergences)
# If you want to save the output to a JSON file:
with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_scaled_grouped_matrices_kl_divergences.json', 'w') as outfile:
    json.dump(kl_divergences, outfile, indent=4)

# Save the KL divergences in a text file
with open('C:/Users/1943069/Scraping_Training_VSC/Training_Exp1/Sample1661_scaled_grouped_matrices_kl_divergences.txt', 'w') as outfile:
    for key, divergence in kl_divergences.items():
        outfile.write(f"Key: {key}\n")
        outfile.write(f"KL Divergence: {divergence:.6f}\n")
        outfile.write("-" * 30 + "\n")  # Separator line for readability