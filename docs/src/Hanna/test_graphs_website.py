import sys
import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import contractions
import numpy as np

# Add the parent directory of Milestone_P2 to sys.path to allow importing utils
# Current script is in src/Hanna
# utils is in src/Milestone_P2/utils
# We need to add src/Milestone_P2 to sys.path to import utils.structural_analysis

# Get the current script's directory
current_dir = Path(__file__).resolve().parent
# Go up to src
src_dir = current_dir.parent
# Path to Milestone_P2
milestone_p2_dir = src_dir / 'Milestone_P2'

sys.path.append(str(milestone_p2_dir))

try:
    import utils.structural_analysis as ut
except ImportError:
    # Fallback if the import fails (e.g. if the structure is different than expected)
    # But based on the file list, this should work.
    print("Could not import utils.structural_analysis. Please ensure the path is correct.")
    sys.exit(1)

def main():
    # PATH PROJECT SETUP
    # Assuming we are in src/Hanna, project root is ../../
    PROJECT_ROOT = current_dir.parent.parent
    FIGURES_DIR = PROJECT_ROOT / 'figures'
    
    # DATA DIRECTORIES
    DATA_DIR = PROJECT_ROOT / 'data' / 'newyorker_caption_contest'
    CSV_CLEAN_DIR = DATA_DIR / 'data_cleaned'
    
    print("Project path setup:")
    print("PROJECT_ROOT:  ", PROJECT_ROOT)
    print("CSV_CLEAN_DIR: ", CSV_CLEAN_DIR)
    
    if not CSV_CLEAN_DIR.exists():
        print(f"WARNING: CSV_CLEAN_DIR does not exist at {CSV_CLEAN_DIR}")
        # Fallback to check if it's just created relative to current working directory if running from root
        # But sticking to the logic requested.
    
    # 4.1 Data handling, cleaning up for the structural analysis
    print("Loading all contest data from cleaned directory...")
    try:
        # Load data using the utility function, but point to CSV_CLEAN_DIR
        df = ut.load_all_contest_data(str(CSV_CLEAN_DIR), start_id=510, end_id=895)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df.empty:
        print("No data loaded. Exiting.")
        return

    print(f"Data loaded. Shape: {df.shape}")

    # Pre-processing (from Cell 59)
    print("Preprocessing captions...")
    # check for missing values and fill NaN w/ empty string
    df["caption"] = df["caption"].fillna("")

    # erase unnecessary punctuations :
    df["caption"] = df["caption"].str.replace(r"[--]", " ", regex=True)
    df["caption"] = df["caption"].str.replace(r"[.,?]", " ", regex=True)

    # lowercase : 
    df["caption"] = df["caption"].str.lower()

    # contractions :
    # applying contractions can be slow on large datasets, using progress bar if possible or just running it
    print("Fixing contractions (this might take a while)...")
    df["caption"] = df["caption"].apply(lambda x: contractions.fix(x))   

    # 4.2 Are longer jokes funnier than shorter ones?
    print("Calculating metrics...")
    # words
    df["word_counts"] = [len(line.split()) for line in df["caption"]]

    # characters 
    df["length"] = [len(line) for line in df["caption"]]

    # average of words and characters
    print(f"avg word count: {df['word_counts'].mean():.2f}")
    print(f"avg character count: {df['length'].mean():.2f}")
    
    # scores
    df['total_votes'] = df['not_funny'] + df['somewhat_funny'] + df['funny'] # this is for each caption 

    # Avoid division by zero
    df = df[df['total_votes'] > 0].copy()

    # weighted average (2 for funny, 1 for somewhat_funny, 0 for not_funny)
    df['funny_score_weighted'] = (df['funny'] * 2 + df['somewhat_funny'] * 1 + df['not_funny'] * 0) / df['total_votes']
    df['success_metric'] = df['funny_score_weighted']

    # correlation analysis 
    corr_word_score, p_word = pearsonr(df['word_counts'], df['success_metric'])
    corr_char_score, p_char = pearsonr(df['length'], df['success_metric'])

    print(f"\ncorrelation w/ funny Score:")
    print(f"# of words correlation: {corr_word_score:.3f} (p-value: {p_word:.3f})")
    print(f"# of characters correlation: {corr_char_score:.3f} (p-value: {p_char:.3f})")

    # Plotting
    print("Generating plots...")
    # style of plot
    plt.style.use('default')
    sns.set_palette("husl")

    df_filtered = df[(df['word_counts'] <= 100) & (df['length'] <= 400)]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df_filtered, x='word_counts', y='success_metric', alpha=0.6)
    plt.title(f'word count vs funny score\n(r: {corr_word_score:.3f})')

    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df_filtered, x='length', y='success_metric', alpha=0.6)
    plt.title(f'character count (= length) vs funny score\n(r: {corr_char_score:.3f})')

    plt.subplot(1, 3, 3)
    # Create performance groups for comparison
    # Check if there are enough unique values to bin
    if df['success_metric'].nunique() > 3:
        df['performance_group'] = pd.cut(df['success_metric'], bins=3, labels=['Low', 'Medium', 'High'])
        sns.boxplot(data=df, x='performance_group', y='word_counts')
        plt.title('word count by performance group')
    else:
        print("Not enough unique success_metric values to create performance groups.")

    plt.tight_layout()
    
    # Save the plot or show it
    # Since this is a script, showing might not work in all environments, but I'll include plt.show()
    # and also save it to the current directory for verification.
    output_plot = current_dir / 'graph_4_2_repro.png'
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    main()

