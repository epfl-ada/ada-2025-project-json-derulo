import os
import csv
import random
from collections import defaultdict
import re
import pandas as pd

input_folder = CSV_DIR
clean_folder = CSV_CLEAN_DIR

# ============================================================================= 
# CHAPTER 2 - Data pre-processing & cleaning
# =============================================================================

# Part 2.1 : Data Wrangling

def has_correct_headers(headers):
    """
    function used to check if headers are right
    Parameters: headers (list): List of CSV headers 
    Returns True if the CSV headers match the expected headers exactly.
    """
    EXPECTED_HEADERS = ["rank", "caption", "mean", "precision", "votes", "not_funny", "somewhat_funny", "funny"]
    return headers == EXPECTED_HEADERS

# Iterate over all CSV files
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_folder, filename)

        with open(input_path, "r", encoding="utf-8", newline='') as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration:
                print(f"{filename}: file is empty")
                continue

            if not has_correct_headers(headers):
                print(f"{filename}: incorrect headers -> {headers}")


#=============================================================================

def fix_missing_rank(file_path,file_added):
    """
    fixing the files with the missing rank header
    Parameters: file_path (str): Path to the CSV file to fix
    Add a rank column based on descending mean values."""
    rows = []

    EXPECTED_HEADERS = ["rank", "caption", "mean", "precision", "votes", "not_funny", "somewhat_funny", "funny"]

    MISSING_RANK_HEADERS = ["caption", "mean", "precision", "votes", "not_funny", "somewhat_funny", "funny"]
    # Read the CSV
    with open(file_path, "r", encoding="utf-8", newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['mean'] = float(row['mean'])  # convert mean to float for sorting
                rows.append(row)
            except ValueError:
                continue  # skip rows with invalid mean

    # Sort rows by mean descending
    rows.sort(key=lambda x: x['mean'], reverse=True)

    # Write back the CSV with rank
    with open(file_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(EXPECTED_HEADERS)  # write header

        for i, row in enumerate(rows):
            writer.writerow([
                i,  # rank
                row['caption'],
                row['mean'],
                row['precision'],
                row['votes'],
                row['not_funny'],
                row['somewhat_funny'],
                row['funny']
            ])

# Check and fix only this file
input_path = os.path.join(input_folder, file_to_fix)

with open(input_path, "r", encoding="utf-8", newline='') as f:
    reader = csv.reader(f)
    try:
        headers = next(reader)
    except StopIteration:
        print(f"{file_to_fix} is empty.")
        headers = []

if headers == MISSING_RANK_HEADERS:
    print(f"{file_to_fix} is missing 'rank'. Fixing it now...")
    fix_missing_rank(input_path)
    print(f"{file_to_fix} has been fixed.")
else:
    print(f"{file_to_fix} already has correct headers.")

#=============================================================================

def check_corruption(line):
    """
    Params: line (str): A line from the CSV file
    Return a list of reasons if the line is corrupted.
    Criteria:
      1. Line must start with a single number (rank)
      2. Line must end with a numeric value (funny score)
      also removes blank spaces, commas etc from checking
    """
    pattern_starts_with_one_number = re.compile(r"^\d+,")
    pattern_numeric_end = re.compile(r"[-+]?\d*\.?\d+$")

    reasons = []
    stripped = line.strip()

    # --- Check start ---
    if not pattern_starts_with_one_number.match(stripped):
        reasons.append("does not start with single number")

    # --- Check end ---
    if stripped.endswith(',') or stripped.endswith('"') or not stripped:
        # empty, dangling quote, or dangling comma
        reasons.append("ends with non-numeric value")
    else:
        last_field = stripped.split(",")[-1].strip()
        if not pattern_numeric_end.match(last_field):
            reasons.append("ends with non-numeric value")

    return reasons

#=====================================================================

def check_csv_file_for_corruption(file_num, folder_path):
    """
    Check a CSV file for corrupted lines, given its file number and folder path.
    Prints corrupted lines and a summary at the end.
    Returns corrupted_line_numbers (list): actual 1-based file line numbers
    """
    filename = f"{file_num}.csv"
    file_path = os.path.join(folder_path, filename)

    if not os.path.isfile(file_path):
        print(f" File not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8", newline='') as f:
        lines = f.readlines()

    found_corruption = False
    corrupted_line_numbers = []  # actual file line numbers (1-based)


    print(f"Checking {filename} for corrupted lines...\n")

    for line_number, line in enumerate(lines, start=1):
        if line_number == 1:
            continue  # skip header
        reasons = check_corruption(line)
        if reasons:
            found_corruption = True
            corrupted_line_numbers.append(line_number)

            print(f"{filename}: line {line_number} -> {', '.join(reasons)}")
            print(f"  → {line.strip()[:150]}{'...' if len(line.strip()) > 150 else ''}")

    if not found_corruption:
        print(f" {filename} has no corrupted lines.\n")
    else:
        print("\n--- Summary ---")
        print(f"Total corrupted lines: {len(corrupted_line_numbers)}")
        print(f"Rows with corruption (file line numbers): {corrupted_line_numbers}\n")


    return corrupted_line_numbers


#=============================================================================

def clean_csv_using_corruption(file_num, folder_path):
    """
    Cleans a CSV file by removing corrupted rows found by check_csv_file_for_corruption.
    Saves the cleaned file in a 'data_cleaned' folder in the same parent directory.
    Uses only the csv module, no pandas.
    """
    # Get corrupted rows (0-based for CSV reading after header)
    corrupted_line_numbers = check_csv_file_for_corruption(file_num, folder_path)



    filename = f"{file_num}.csv"
    file_path = os.path.join(folder_path, filename)

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    # Prepare clean folder
    parent_dir = os.path.dirname(folder_path)
    clean_folder = os.path.join(parent_dir, "data_cleaned")
    os.makedirs(clean_folder, exist_ok=True)
    cleaned_filename = f"{file_num}_cleaned.csv"
    cleaned_path = os.path.join(clean_folder, cleaned_filename)

    # Read and write CSV, skipping corrupted rows
    with open(file_path, "r", newline="", encoding="utf-8") as infile, \
         open(cleaned_path, "w", newline="", encoding="utf-8") as outfile:

        

        for line_number, line in enumerate(infile, start=1):
            if line_number == 1 or line_number not in corrupted_line_numbers:
                outfile.write(line)

                
                
    return cleaned_path, corrupted_line_numbers
#=============================================================================

def check_cleaned_csv_for_corruption(clean_folder):
    """
    Checks all CSV files in the cleaned folder for corruption.

    This function iterates over each CSV file in `clean_folder` and uses
    the helper function `check_csv_file_for_corruption` to detect any corrupted lines.
    It prints a summary indicating whether all files are clean or if some contain corrupted lines.

    Parameters:
        clean_folder (str): Path to the folder containing cleaned CSV files.

    Returns:
        bool: True if all CSV files are clean (no corrupted lines), False if any files contain corruption.
    """
    all_clean = True  # flag to track if everything is clean

    for filename in os.listdir(clean_folder):
        if not filename.endswith(".csv"):
            continue

        # The file prefix is already file_num_cleaned
        file_num_cleaned = filename.replace(".csv", "")

        # Check corruption
        corrupted_lines = check_csv_file_for_corruption(file_num_cleaned, clean_folder)
        if corrupted_lines:
            all_clean = False

    # Final summary
    if all_clean:
        print("\n✅ All files in cleaned_input_folder are clean (no corrupted lines).")
    else:
        print("\n❌ Some files have corrupted lines. Check the output above for details.")

    return all_clean
