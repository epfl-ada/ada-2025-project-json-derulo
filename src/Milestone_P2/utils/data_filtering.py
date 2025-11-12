import os
import csv
import random
from collections import defaultdict
import re
import pandas as pd

input_folder = CSV_DIR

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
                row['mean'] = float(row['mean'])  
                rows.append(row)
            except ValueError:
                continue  

    # Sort rows by mean descending
    rows.sort(key=lambda x: x['mean'], reverse=True)

    # Write back the CSV with rank
    with open(file_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(EXPECTED_HEADERS)  
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
file_to_fix = file_added
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
    """
    pattern_starts_with_one_number = re.compile(r"^\d+,")  # should start with a single number
    pattern_non_numeric_end = re.compile(r"[^\d.,-]$")    # should end with a number

    reasons = []
    stripped = line.strip()

    # Check start
    if not pattern_starts_with_one_number.match(stripped):
        reasons.append("does not start with single number")

    # Check end
    if pattern_non_numeric_end.search(stripped.split(",")[-1]):
        reasons.append("ends with non-numeric value")

    return reasons

#=====================================================================

def check_csv_file_for_corruption(file_num, folder_path):
    """
    Check a CSV file for corrupted lines, given its file number and folder path.
    Prints corrupted lines and a summary at the end.
    Returns a list of row indices (0-based for pandas) that are corrupted.
    """
    filename = f"{file_num}.csv"
    file_path = os.path.join(folder_path, filename)

    if not os.path.isfile(file_path):
        print(f" File not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8", newline='') as f:
        lines = f.readlines()

    found_corruption = False
    corrupted_rows = []  

    print(f"Checking {filename} for corrupted lines...\n")

    for line_number, line in enumerate(lines, start=1):
        if line_number == 1:
            continue  # skip header
        reasons = check_corruption(line)
        if reasons:
            found_corruption = True
            corrupted_rows.append(line_number - 2)  
            print(f"{filename}: line {line_number} -> {', '.join(reasons)}")
            print(f"  → {line.strip()[:150]}{'...' if len(line.strip()) > 150 else ''}")

    if not found_corruption:
        print(f" {filename} has no corrupted lines.\n")
    else:
        print("\n--- Summary ---")
        print(f"Total corrupted lines: {len(corrupted_rows)}")
        print(f"Rows with corruption: {[r+2 for r in corrupted_rows]}\n")  

    return corrupted_rows

#=============================================================================

def clean_csv_using_corruption(file_num, folder_path):
    """
    Cleans a CSV file by removing corrupted rows found by check_csv_file_for_corruption.
    Saves the cleaned file in a 'data_cleaned' folder in the same parent directory.
    Uses only the csv module, no pandas.
    """
    # Get corrupted rows (0-based for CSV reading after header)
    corrupted_rows = check_csv_file_for_corruption(file_num, folder_path)

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

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row_idx, row in enumerate(reader):
            if row_idx == 0:
                # Always write header
                writer.writerow(row)
            elif (row_idx - 1) not in corrupted_rows:
                # Only write non-corrupted rows
                writer.writerow(row)

    print(f"{filename}: Cleaned file saved to {cleaned_path}, removed {len(corrupted_rows)} corrupted rows.\n")