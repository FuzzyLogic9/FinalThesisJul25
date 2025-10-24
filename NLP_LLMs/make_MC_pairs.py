import pandas as pd

def clean_file(input_file, output_file):
    # Determine file type
    file_ext = input_file.split('.')[-1]

    # === Load file based on extension ===
    if file_ext == 'csv':
        df = pd.read_csv(input_file, dtype=str)
    elif file_ext in ['xls', 'xlsx']:
        df = pd.read_excel(input_file, dtype=str)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    # === Step 1: Use only the first and last columns ===
    first_col = df.columns[0]
    last_col = df.columns[-1]
    df = df[[first_col, last_col]]
    df.columns = ['PageID', 'AddTox']  # Rename columns

    # === Step 2: Convert AddTox to numeric (cleaning) ===
    df['AddTox'] = pd.to_numeric(df['AddTox'], errors='coerce').fillna(0).astype(str)

    # === Step 3: Generate Pairs for MC ===
    df['Pairs for MC'] = df['AddTox'] + ',' + df['AddTox'].shift(-1).fillna('')

    # === Step 4: Forward-fill PageID if needed ===
    df['PageID'] = df['PageID'].ffill()

    # === Step 5: Keep only the two required columns ===
    final_df = df[['PageID', 'Pairs for MC']]

    # === Step 6: Save the result ===
    if file_ext == 'csv':
        final_df.to_csv(output_file, index=False)
    elif file_ext in ['xls', 'xlsx']:
        final_df.to_excel(output_file, index=False)

    print(f"✅ Cleaned file saved as: {output_file}")

# === Example usage ===
base_path = "C:/Users/fzkar/final_code_data/GPT_outputs/GPT_Outputs/"
file1_path = base_path + "Final_GPTNEO_FT_G1_OT_predicted.csv"
file2_path = base_path + "Page_Paired_GPTNEO_FT_G1_OT.csv"

clean_file(file1_path, file2_path)
