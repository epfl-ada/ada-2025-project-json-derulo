# This file contains all the relevant functions used in the 'Milestone_P2.ipynb' file for the structural analysis.

## STRUCTURAL ANALYSIS FUNCTION

#### Necessary libraries
import json
import pandas as pd
import os
from datetime import datetime, timedelta

# ============================================================================= 
# CHAPTER 2 - Data pre-processing & cleaning
# =============================================================================

# Part 2.1 : Data Wrangling

def check_votes(json_df, folder):
    """
    Goes through all contests in json_df ,
    computes total votes from corresponding CSVs, and compares with num_votes
    in metadata of the json_df.

    Parameters: json_df (pd.DataFrame): DataFrame containing contest data with 'contest_id' and 'metadata' dictionary
                csv_folder (str): Path to the folder containing CSV files
    
    Returns a DataFrame with the results.
    Also prints a summary of matches vs mismatches.
    """
    results = []

    for _, row in json_df.iterrows():
        contest_id = str(row["contest_id"])
        metadata = row.get("metadata", {})

        # ensure metadata is a dict (in case it's NaN or string)
        if not isinstance(metadata, dict):
            metadata = {}

        csv_path = folder / f"{contest_id}.csv"
        try:
            df = pd.read_csv(csv_path)

            # compute total votes
            total_votes = pd.to_numeric(df["votes"], errors="coerce").sum()
            json_votes = metadata.get("num_votes", None)

            # compare (only if json_votes exists)
            match = (json_votes is not None) and (int(total_votes) == int(json_votes))

            results.append({
                "contest_id": contest_id,
                "csv_found": True,
                "csv_votes": int(total_votes),
                "json_votes": json_votes,
                "match": match
            })

        except Exception as e:
            results.append({
                "contest_id": contest_id,
                "csv_found": True,
                "csv_votes": None,
                "json_votes": metadata.get("num_votes", None),
                "match": None,
                "error": str(e)
            })

    df_results = pd.DataFrame(results)

    # summary print
    total = len(df_results)
    matches = df_results["match"].sum(skipna=True)
    print(f"Total contests checked: {total}")
    print(f"Matching vote counts: {int(matches)}")
    print(f"Mismatches or missing: {total - int(matches)}")

    return df_results


# ============================================================================= 

#Part 2.2 : Data Enrichment:

def update_metadata_from_csv(df, folder):
    """
    Goes through all contests in json_df ,
    adds 'num_captions' and 'num_votes' to missing metadata

    Parameters: json_df (pd.DataFrame): DataFrame containing contest data with 'contest_id' and 'metadata' dictionary
    the contest that don't have metadata have 
                csv_folder (str): Path to the folder containing CSV files
    
    Returns a DataFrame with the added metadata.
    """    
    for i, row in df.iterrows():
        # Ensure metadata is a dict
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            meta = {}

        # Skip if both fields already exist
        need_caps = ("num_captions" not in meta) 
        need_votes = ("num_votes" not in meta) 
        if not (need_caps or need_votes):
            df.at[i, "metadata"] = meta
            continue

        # Build CSV path from the 'data' field
        data_field = row.get("data")
        if not isinstance(data_field, str) :
            df.at[i, "metadata"] = meta
            continue

        filename = os.path.basename(data_field)      
        csv_path = os.path.join(folder, filename) 
        # If file not found, skip it
        if not os.path.exists(csv_path):
            df.at[i, "metadata"] = meta
            continue
        # Read CSV and compute values
        try:
            cdf = pd.read_csv(csv_path)
            meta["num_captions"] = int(len(cdf))
            meta["num_votes"] = int(cdf["votes"].sum(skipna=True))
        except Exception as e:
            pass  
        # Write metadata dict back into the DataFrame
        df.at[i, "metadata"] = meta
        
    return df


# =============================================================================

def add_contest_dates(df):
    """
    Adds contest_day_publish and contest_day_votes to each row's metadata.
    Uses contest 510 as base (22 Feb 2016).
    Parameters: df (pd.DataFrame): DataFrame containing enriched json dataframe
    Returns: pd.DataFrame with added contest dates in metadata.
    """
    contest_510_publish = datetime.strptime('2016.02.22', "%Y.%m.%d")
    contest_510_votes = datetime.strptime('2016.02.29', "%Y.%m.%d")

    for i in range(len(df)):
        contest_id = df['contest_id'].iloc[i]
        diff = int(contest_id) - 510
        contest_day_publish = contest_510_publish + timedelta(days=diff * 7)

        # your original assignments
        df['metadata'].iloc[i]['contest_day_publish'] = contest_day_publish
        df['metadata'].iloc[i]['contest_day_votes'] = contest_day_publish + timedelta(days=7)

    return df

