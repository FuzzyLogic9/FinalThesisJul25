import pandas as pd
import os

# Set base path
base_path = r'C:\FuzzyNov2024\toxicity_project\final_code_data'

# CSV filenames
csv_files = [
    'tox_matching_first_reviews_only.csv',
    'att_matching_first_reviews_only.csv',
    'agg_matching_first_reviews_only.csv'  # assuming the correct third file
]

# Load PageID groups
def load_ids(filepath):
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f if line.strip().isdigit())

group_one_ids = load_ids(os.path.join(base_path, 'ids_group_one_pageids.txt'))
group_zero_ids = load_ids(os.path.join(base_path, 'ids_group_zero_pageids.txt'))

# Load and combine CSVs
all_reviews = pd.DataFrame()
for file in csv_files:
    df = pd.read_csv(os.path.join(base_path, file))
    all_reviews = pd.concat([all_reviews, df], ignore_index=True)

# Standardize column names
all_reviews.columns = [col.strip().lower() for col in all_reviews.columns]

# Ensure correct columns
required_cols = {'page_id', 'rev_id', 'cleaned_comment'}
if not required_cols.issubset(all_reviews.columns):
    raise ValueError(f"Missing one of the required columns: {required_cols}")

# Convert page_id to string
all_reviews['page_id'] = all_reviews['page_id'].astype(str)

# Drop duplicate rev_ids
all_reviews = all_reviews.drop_duplicates(subset='rev_id', keep='first')

# Filter by group and add label
group_one_df = all_reviews[all_reviews['page_id'].isin(group_one_ids)].copy()
group_one_df['group_label'] = 1

group_zero_df = all_reviews[all_reviews['page_id'].isin(group_zero_ids)].copy()
group_zero_df['group_label'] = 0

# Save separate group files
group_one_df.to_csv(os.path.join(base_path, 'group_one_reviews.csv'), index=False)
group_zero_df.to_csv(os.path.join(base_path, 'group_zero_reviews.csv'), index=False)

# Merge and save final dataset
merged_df = pd.concat([group_one_df, group_zero_df], ignore_index=True)
merged_df.to_csv(os.path.join(base_path, 'merged_reviews_with_labels.csv'), index=False)

# Print summary
print(f"✅ Group One (after dedup): {len(group_one_df)} rows")
print(f"✅ Group Zero (after dedup): {len(group_zero_df)} rows")
print(f"✅ Total (deduplicated): {len(merged_df)} rows (Expected: 1662)")
