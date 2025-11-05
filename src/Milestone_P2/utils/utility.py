# This file contains all the relevant functions used in the 'Milestone_P2.ipynb' file.

## STRUCTURAL ANALYSIS FUNCTION

#### Necessary libraries
import pandas as pd
from ipywidgets import interact, IntSlider
from IPython.display import display, Markdown, Image
from pathlib import Path
import string
import re
import contractions
import seaborn as sns
import scipy
from scipy.stats import pearsonr, spearmanr, ttest_ind, pearsonr, mannwhitneyu
import sys 
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

### Function to fetch csv files (being able to choose which one/how many to analyse)

def load_all_contest_data(base_path, start_id=510, end_id=895):
    """
    Load CSV files and combine into one DataFrame 
    
    -> possibility of not having to read everything: 
            df_510_520 = load_all_contest_data(base_path, start_id=510, end_id=520) (example)
    """
    base_path = Path(base_path)
    all_dfs = []
    missing_files = []
    
    for contest_id in range(start_id, end_id + 1):
        file_path = base_path / f"{contest_id}.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                df['contest_id'] = contest_id
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            missing_files.append(contest_id)
    
    if missing_files:
        print(f"Missing files for contest IDs: {missing_files}")
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Successfully loaded {len(all_dfs)} files")
        print(f"Total rows: {len(combined_df)}")
        return combined_df
    else:
        print("no files were loaded")
        return pd.DataFrame()


### Functions used to get rid of common words such as "a, the, an, he, she, I..." so that we don't cound them in (these are considered stop words)

def get_stopword_lists():
    '''This function is responsible of getting stop word lists'''
    nltk.download('stopwords')
    basic_stopwords = set(stopwords.words('english'))
    
    # extended stop words (added them ourselves after observation)
    extended_stopwords = basic_stopwords.union({
        'like', 'get', 'would', 'could', 'also', 'us', 'one', 'two', 'three'
    })
    
    # but keep negation words
    sentiment_safe_stopwords = basic_stopwords - {
        'not', 'no', 'nor', 'never', 'none', 'neither', 'do not',  
        'are not', 'could not', 'did not',
        'does not', 'had not', 'has not', 'have not', 'is not', 'might not', 'must not',
        'need not', 'shall not', 'should not', 
        'was not', 'were not', 'will not', 'would not'
    }
    return {
        'basic': basic_stopwords,
        'extended': extended_stopwords,
        'sentiment_safe': sentiment_safe_stopwords,
        'custom': set()  
    }

def analyze_with_stopwords(df, caption_column='caption', stopword_type='basic', top_n=20):
    '''This function is responsible of analysing the word frequency w/ the different kinds of stopwords'''
    stopword_lists = get_stopword_lists()
    selected_stopwords = stopword_lists[stopword_type]
    
    # combine
    all_text = ' '.join(df[caption_column].dropna().astype(str))
    
    words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
    
    # remove stopwords
    if selected_stopwords:
        words = [word for word in words if word not in selected_stopwords]
    
    word_freq = Counter(words)
    
    return pd.DataFrame(word_freq.most_common(top_n), 
                       columns=['word', 'frequency'])

### Function that gets most common words

def get_most_common_words(text_series, n=30, remove_stopwords=True):
    # Combine all captions into one text
    all_text = ' '.join(text_series.dropna().astype(str))
    
    # Clean and tokenize text
    words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
    
    # Remove stopwords (optional)
    if remove_stopwords:
        from nltk.corpus import stopwords
        import nltk
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
    
    # Count word frequencies
    word_counts = Counter(words)
    return word_counts.most_common(n)

### Calculation of vocabulary richness 

def calculate_vocabulary_richness(text_series, remove_stopwords=True):
    """
    calculation of vocabulary richness metrics for a series of texts
    Returns: dataFrame w/ TTR, unique_words, total_words for each caption
    """
    results = []
    stop_words = set()
    if remove_stopwords:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
    
    for caption in text_series:
        if pd.isna(caption) or caption == '':
            results.append({'ttr': 0, 'unique_words': 0, 'total_words': 0})
            continue
        words = re.findall(r'\b[a-zA-Z]+\b', caption.lower())
        
        if remove_stopwords:
            words = [word for word in words if word not in stop_words]
        
        if len(words) == 0:
            results.append({'ttr': 0, 'unique_words': 0, 'total_words': 0})
            continue
            
        # metrics
        unique_words = len(set(words))
        total_words = len(words)
        ttr = unique_words / total_words if total_words > 0 else 0
        
        results.append({
            'ttr': ttr,
            'unique_words': unique_words,
            'total_words': total_words
        })
    
    return pd.DataFrame(results)