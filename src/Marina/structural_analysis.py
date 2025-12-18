# This file contains all the relevant functions used in the 'Milestone_P2.ipynb' file for the structural analysis.

## STRUCTURAL ANALYSIS FUNCTION

import pandas as pd
from pathlib import Path
import re
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
# import string
# import contractions
# from ipywidgets import interact, IntSlider
# from IPython.display import display, Markdown, Image
# import scipy
# import utils.structural_analysis as ut
# from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu 
# import sys 
# from sklearn.feature_extraction.text import TfidfVectorizer

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
        file_path = base_path / f"{contest_id}_cleaned.csv"
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

def remove_stopwords(caption, stopword_type='basic'):
    '''erase stopwords from a single caption'''
    if pd.isna(caption) or not isinstance(caption, str):
        return ""
    
    stopword_lists = get_stopword_lists()
    selected_stopwords = stopword_lists[stopword_type]

    words = re.findall(r'\b[a-zA-Z]+\b', caption.lower())
    if selected_stopwords:
        words = [word for word in words if word not in selected_stopwords]
    
    # cleaned caption
    return ' '.join(words)

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

## Plotting function for TTR

def plot_linguistic_metrics(df, top_captions, bottom_captions, corr_ttr, corr_unique, 
                           metrics=['ttr', 'unique_words'], 
                           performance_group_col='performance_group',
                           success_metric_col='success_metric'):
    
    # scatter plots and boxplot
    plt.figure(figsize=(15, 5))
    
            # scatter plots for each metric
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.scatterplot(data=df, x=metric, y=success_metric_col, alpha=0.6)
        
        corr = corr_ttr if metric == 'ttr' else corr_unique
        metric_name = 'TTR' if metric == 'ttr' else 'Unique Words'
        
        plt.title(f'{metric_name} vs Funny Score\n(r: {corr:.3f})')
        plt.xlabel(metric_name)
        plt.ylabel('Funny Score')
    
            # boxplot for TTR / performance group
    plt.subplot(1, 3, 3)
    sns.boxplot(data=df, x=performance_group_col, y='ttr')
    plt.title('TTR by Performance Group')
    plt.xlabel('Performance Group')
    plt.ylabel('Type-Token Ratio')
    
    plt.tight_layout()
    plt.show()
    
    # TTR distribution and examples
    plt.figure(figsize=(12, 4))
    
    # TTR distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data=top_captions, x='ttr', bins=20, alpha=0.7, label='Top Performers', color='green')
    sns.histplot(data=bottom_captions, x='ttr', bins=20, alpha=0.7, label='Bottom Performers', color='red')
    plt.title('TTR Distribution: Top vs Bottom Performers')
    plt.xlabel('Type-Token Ratio')
    plt.ylabel('Frequency')
    plt.legend()
    
    # High vs Low TTR examples
    plt.subplot(1, 2, 2)
    high_ttr_captions = df.nlargest(5, 'ttr')[['caption', 'ttr', success_metric_col]]
    low_ttr_captions = df.nsmallest(5, 'ttr')[['caption', 'ttr', success_metric_col]]
    
    examples_df = pd.concat([
        high_ttr_captions.assign(type='High TTR'),
        low_ttr_captions.assign(type='Low TTR')
    ])
    
    sns.scatterplot(data=examples_df, x='ttr', y=success_metric_col, hue='type', s=100)
    plt.title('High vs Low TTR Captions\n(Extreme Examples)')
    plt.xlabel('Type-Token Ratio')
    plt.ylabel('Funny Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

