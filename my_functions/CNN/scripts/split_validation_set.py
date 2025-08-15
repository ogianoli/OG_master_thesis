import pandas as pd
import random

# --- CONFIG ---
source_csv = "master_thesis/my_functions/CNN/data/functions.csv"
validation_csv = "master_thesis/my_functions/CNN/data/validation_functions.csv"
filtered_csv = "master_thesis/my_functions/CNN/data/functions_train_test.csv"  # Remaining data (used for train/test)

# --- Load all rows ---
with open(source_csv, 'r') as f:
    lines = f.readlines()

# --- Randomly select 500 rows for validation ---
validation_indices = random.sample(range(len(lines)), 500)
validation_lines = [lines[i] for i in validation_indices]

# --- Remaining data for train/test ---
remaining_lines = [line for i, line in enumerate(lines) if i not in validation_indices]

# --- Save both ---
with open(validation_csv, 'w') as f:
    f.writelines(validation_lines)

with open(filtered_csv, 'w') as f:
    f.writelines(remaining_lines)

print(f"✅ Saved validation set to {validation_csv}")
print(f"✅ Remaining data saved to {filtered_csv}")