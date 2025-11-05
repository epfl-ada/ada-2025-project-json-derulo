#code used to clean-up the cpations which are not well formatted
#the detection criteria is : missing rank + 3 consecutive digits separated by commas
import os
import csv

#folder_path = r"C:\Users\hanna\ADA\2025\newyorker_caption_contest\data"
#C:\Users\hanna\ADA\2025\newyorker_caption_contest\data

import os
import csv

input_folder = r"C:\Users\hanna\ADA\2025\newyorker_caption_contest\data"
MAX_CAPTION_LENGTH = 250  # threshold for corrupted captions

def is_corrupted_row(row, caption_index=1, max_length=MAX_CAPTION_LENGTH):
    """
    Detect corrupted rows:
    - Rows with fewer than 8 columns (malformed)
    - Rows with caption length > max_length
    """
    if len(row) != 8:
        return True  # malformed row
    caption = row[caption_index]
    return len(caption) > max_length

# Iterate over all CSV files
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_folder, filename)

        with open(input_path, "r", encoding="utf-8", newline='') as infile:
            reader = csv.reader(infile)
            next(reader)  # skip header

            for row_number, row in enumerate(reader, start=2):
                if is_corrupted_row(row):
                    print(f"{filename}: corrupted row at line {row_number}")