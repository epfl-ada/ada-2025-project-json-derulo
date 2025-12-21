import os
import csv
import random
from collections import defaultdict
import re
import pandas as pd
from io import StringIO

input_folder = "../../../data/newyorker_caption_contest/data"
cleaned_input_folder =  "../../../data/newyorker_caption_contest/data_cleaned"

#function used to check if headers are right
EXPECTED_HEADERS = ["rank", "caption", "mean", "precision", "votes", "not_funny", "somewhat_funny", "funny"]

def has_correct_headers(headers):
    """
    Returns True if the CSV headers match the expected headers exactly.
    """
    return headers == EXPECTED_HEADERS

MISSING_RANK_HEADERS = ["caption", "mean", "precision", "votes", "not_funny", "somewhat_funny", "funny"]

def fix_missing_rank(file_path):
    """Add a rank column based on descending mean values."""
    rows = []

    # Read the CSV
    with open(file_path, "r", encoding="utf-8", newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['mean'] = float(row['mean'])  # convert mean to float for sorting
                rows.append(row)
            except ValueError:
                continue  # skip rows with invalid mean
            
def fix_missing_rank(file_path):
    """Add a rank column based on descending mean values."""
    rows = []

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

def check_corruption(line):
    """
    Params: line (str): A line from the CSV file
    Return a list of reasons if the line is corrupted.
    Criteria:
      1. Line must start with a single number (rank)
      2. Line must end with a numeric value (funny score)
      3. Line must parse into exactly 8 CSV fields
    """
    pattern_starts_with_one_number = re.compile(r"^\d+,")
    pattern_numeric_end = re.compile(r"[-+]?\d*\.?\d+$")

    reasons = []
    stripped = line.strip()

    # --- Check start ---
    if not pattern_starts_with_one_number.match(stripped):
        reasons.append("does not start with single number")

    # --- Check CSV structure (added rule) ---
    try:
        parsed = next(csv.reader(StringIO(line)))
        if len(parsed) != 8:
            reasons.append(f"wrong number of columns ({len(parsed)} instead of 8)")
    except Exception:
        reasons.append("CSV parsing error (malformed quotes)")

    # --- Check end ---
    if stripped.endswith(',') or stripped.endswith('"') or not stripped:
        reasons.append("ends with non-numeric value")
    else:
        last_field = stripped.split(",")[-1].strip()
        if not pattern_numeric_end.match(last_field):
            reasons.append("ends with non-numeric value")

    return reasons

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

                
    print(f"{filename}: Cleaned file saved to {cleaned_path}, removed {len(corrupted_line_numbers)} corrupted rows.\n")            
    return cleaned_path, corrupted_line_numbers

    
