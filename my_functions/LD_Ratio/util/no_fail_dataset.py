import csv
import os

curves_bates_1_csv = "master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs/DOE_data/pressure_curves_train.csv"
curves_bates_2_csv = "master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs/DOE_data/pressure_curves_test.csv"
curves_star_csv = "master_thesis/my_tests/10_test_classification_doe/star/DOE_outputs/pressure_curves.csv"
curves_endburner_csv = "master_thesis/my_tests/10_test_classification_doe/endburner/DOE_outputs/pressure_curves.csv"
input_csv_bates_1 = "master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs/DOE_data/params_train.csv"
input_csv_bates_2 = "master_thesis/my_tests/10_test_classification_doe/bates/DOE_outputs/DOE_data/params_test.csv"
input_csv_star = "master_thesis/my_tests/10_test_classification_doe/star/DOE_outputs/params.csv"
input_csv_endburner = "master_thesis/my_tests/10_test_classification_doe/endburner/DOE_outputs/params.csv"


output_csv_no_fail_bates = "master_thesis/my_functions/LD_Ratio/util/no_fail_dataset/no_fail_dataset_bates.csv"
output_csv_no_fail_star = "master_thesis/my_functions/LD_Ratio/util/no_fail_dataset/no_fail_dataset_star.csv"
output_csv_no_fail_endburner = "master_thesis/my_functions/LD_Ratio/util/no_fail_dataset/no_fail_dataset_endburner.csv"

os.makedirs(os.path.dirname(output_csv_no_fail_bates), exist_ok=True)
os.makedirs(os.path.dirname(output_csv_no_fail_star), exist_ok=True)
os.makedirs(os.path.dirname(output_csv_no_fail_endburner), exist_ok=True)

def is_nan(value):
    return value.strip() == '' or value.strip().lower() == 'nan'
def is_comma(value):
    return value.strip() == '' or value.strip() == ','

with open(curves_bates_1_csv, 'r', newline='') as curves_infile, \
        open(input_csv_bates_1, 'r') as params_infile, \
        open(output_csv_no_fail_bates, 'w', newline='') as outfile:
    
    curves_reader = csv.reader(curves_infile)
    params_reader = csv.reader(params_infile)
    writer = csv.writer(outfile)

    header = next(params_reader)  # Skip header row in params
    writer.writerow(header)

    for params_dict, row in zip(params_reader, curves_reader):
        # Skip completely empty lines
        if not row:
            continue

        # Check if first three entries are all NaN
        first_three = row[:3]
        if len(first_three) < 3 or all(is_comma(cell) for cell in first_three):
            continue  # Skip this row

        # Write to new CSV
        writer.writerow(params_dict)
with open(curves_bates_2_csv, 'r', newline='') as curves_infile, \
        open(input_csv_bates_2, 'r') as params_infile, \
        open(output_csv_no_fail_bates, 'a', newline='') as outfile:
    
    curves_reader = csv.reader(curves_infile)
    params_reader = csv.reader(params_infile)
    writer = csv.writer(outfile)


    next(params_reader)  # Skip header row in params

    for params_dict, row in zip(params_reader, curves_reader):
        # Skip completely empty lines
        if not row:
            continue

        # Check if first three entries are all NaN
        first_three = row[:3]
        if len(first_three) < 3 or all(is_comma(cell) for cell in first_three):
            continue  # Skip this row

        # Write to new CSV
        writer.writerow(params_dict)

with open(curves_star_csv, 'r', newline='') as curves_infile, \
        open(input_csv_star, 'r') as params_infile, \
        open(output_csv_no_fail_star, 'w', newline='') as outfile:
    
    curves_reader = csv.reader(curves_infile)
    params_reader = csv.reader(params_infile)
    writer = csv.writer(outfile)

    header = next(params_reader)  # Skip header row in params
    writer.writerow(header)

    for params_dict, row in zip(params_reader, curves_reader):
        # Skip completely empty lines
        if not row:
            continue

        # Check if first three entries are all NaN
        first_three = row[:3]
        if len(first_three) < 3 or all(is_nan(cell) for cell in first_three):
            continue  # Skip this row

        # Write to new CSV
        writer.writerow(params_dict)

with open(curves_endburner_csv, 'r', newline='') as curves_infile, \
        open(input_csv_endburner, 'r') as params_infile, \
        open(output_csv_no_fail_endburner, 'w', newline='') as outfile:
    
    curves_reader = csv.reader(curves_infile)
    params_reader = csv.reader(params_infile)
    writer = csv.writer(outfile)

    header = next(params_reader)  # Skip header row in params
    writer.writerow(header)

    for params_dict, row in zip(params_reader, curves_reader):
        # Skip completely empty lines
        if not row:
            continue

        # Check if first three entries are all NaN
        first_three = row[:3]
        if len(first_three) < 3 or all(is_nan(cell) for cell in first_three):
            continue  # Skip this row

        # Write to new CSV
        writer.writerow(params_dict)