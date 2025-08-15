# import pandas as pd
# import numpy as np

# # --- CONFIG: Change this ---
# original_csv_path = "/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs/pressure_curves.csv"
# cleaned_csv_path = "/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_functions/CNN/data/functions_cleaned_bates.csv"

# # --- Load the CSV ---
# df = pd.read_csv(original_csv_path, header=None)

# # --- Remove rows where first three entries are all NaN ---
# mask = ~(df.iloc[:, :3].isna().all(axis=1))
# df_cleaned = df[mask].copy()

# # --- Add a value "0" to the end of each row ---
# df_cleaned[len(df_cleaned.columns)] = 0

# # --- Save to new path ---
# df_cleaned.to_csv(cleaned_csv_path, index=False, header=False)

# print(f"Saved cleaned CSV to: {cleaned_csv_path}")

import csv
import os


# --- CONFIG: Change this ---
shape = "endburner"  # Change to "bates", "star", or "endburner"

number_to_append = 0
if shape == "bates":
    number_to_append = 0  # For bates, append 0
elif shape == "star":
    number_to_append = 1
elif shape == "endburner":
    number_to_append = 2

# --- Automatic paths ---
original_csv_path = f"/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_tests/10_test_classification_doe/{shape}/DOE_outputs/pressure_curves.csv"
cleaned_csv_path = f"/Users/oliviergianoli/Library/Mobile Documents/com~apple~CloudDocs/Desktop_14/Importante/ETH_Master/01_Thesis/master_thesis/my_functions/CNN/data/functions_cleaned_{shape}.csv"

def is_nan(value):
    return value.strip() == '' or value.strip().lower() == 'nan'

with open(original_csv_path, 'r') as infile, open(cleaned_csv_path, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # Skip completely empty lines
        if not row:
            continue

        # Check if first three entries are all NaN
        first_three = row[:3]
        if len(first_three) < 3 or all(is_nan(cell) for cell in first_three):
            continue  # Skip this row

        # Append "0" as a string or integer — change if needed
        row.append(number_to_append)

        # Write to new CSV
        writer.writerow(row)

print(f"Saved cleaned CSV to: {cleaned_csv_path}")