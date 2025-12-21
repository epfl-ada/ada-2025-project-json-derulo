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

import os 
import json 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from IPython.display import display 
import contractions 
from scipy.stats import pearsonr, ttest_ind, f_oneway 
from statsmodels.stats.multicomp import pairwise_tukeyhsd
nltk.download('stopwords')

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import statsmodels.api as sm 

# Define your color palette
COLORS = {
    'primary': '#FFD200',     
    'secondary': '#2C3E50',    
    'accent1': '#E74C3C',     
    'accent2': '#2980B9',   
    'accent3': '#27AE60',     
    'accent4': '#8E44AD'       
}

# Font settings
FONT_FAMILY = "Playfair Display"

def load_docs_with_theme_csv(docs_csv_path, replace_na=False, debug=False):
    """
    Load the prebuilt docs_with_themes.csv file into a pandas DataFrame
    so you don’t have to rebuild docs from individual contest files. 
    
    Param:
        - docs_csv_path: str to the docs_with_themes.csv file
        - replace_na: If True, replace NaN values in 'text' and 'clean_text' columns with empty strings
        - debug: If True, only load the first 5000 rows for testing purposes
        
    Returns:
        - df: pandas DataFrame with the contents of docs_with_themes.csv
    """
    docs_csv_path = Path(docs_csv_path)
    if not docs_csv_path.exists():
        raise FileNotFoundError(f"docs_with_themes.csv not found at {docs_csv_path}")

    expected_cols = [
        "doc_id", "contest_id", "kind", "text", "clean_text",
        "rank", "mean", "precision", "votes", "not_funny", "somewhat_funny", "funny",
        "theme_id", "theme_label"
    ]
    df = pd.read_csv(docs_csv_path)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"docs_with_themes.csv is missing expected columns: {missing}")
    
    # replace rows with NAN with empty string "" (the embedding step cannot handle those)
    if replace_na:
        df["text"] = df["text"].fillna("")
        df["clean_text"] = df["clean_text"].fillna("")
    
    # keep only expected first 1000 rows for debugging
    if debug:
        df = df.head(5000)

    return df

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
    
    df = df.copy()
    
    # create performance groups if not present
    if performance_group_col not in df.columns:
        df[performance_group_col] = pd.qcut(df[success_metric_col], q=3, labels=['Low', 'Medium', 'High'])
    
    # scatter plots and boxplot
    plt.figure(figsize=(15, 5))
    
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
    high_ttr_captions = df.nlargest(5, 'ttr')[['text', 'ttr', success_metric_col]]
    low_ttr_captions = df.nsmallest(5, 'ttr')[['text', 'ttr', success_metric_col]]
    
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



def proportion_calculation(data, verbose=True):
    ''' This function calculates the vote proportions and weighted
        funny scores for each caption
        
        Parameters: 
            - data = pandas DataFrame
            - verbose = bool, default = True

        Returns: 
            data frame with the following added columns: 'total_votes', 
            'prop_not_funny', 'prop_somewhat_funny', 'prop_funny', 'funny_score_weighted', 'success_metric'
    '''
    data['total_votes'] = data['not_funny'] + data['somewhat_funny'] + data['funny'] # this is for each caption 
    # replace the 0s w/ 1s to avoid any divisions by 0
    data['total_votes'] = data['total_votes'].replace(0,1)

    # proportions for each category
    data['prop_not_funny'] = data['not_funny'] / data['total_votes']
    data['prop_somewhat_funny'] = data['somewhat_funny'] / data['total_votes']
    data['prop_funny'] = data['funny'] / data['total_votes']

    # weighted average (2 for funny, 1 for somewhat_funny, 0 for not_funny)
    data['funny_score_weighted'] = (data['funny'] * 2 + data['somewhat_funny'] * 1 + data['not_funny'] * 0) / data['total_votes']
    data['success_metric'] = data['funny_score_weighted']
    data['success_metric'].replace([np.inf, -np.inf], np.nan, inplace=True)

    data['success_metric'] = data['success_metric'].replace([np.inf, -np.inf], np.nan)

    data = data.dropna(subset=['word_counts', 'length', 'success_metric'])

    return data

def interpretation_correlation(r): 
    if abs(r) >= 0.7:
        return "Strong"
    elif abs(r) >= 0.3:
        return "Moderate"
    elif abs(r) >= 0.1:
        return "Weak"
    else:
        return "Very weak"

def plot_prop_step_one(data, data_filtered, word_col, char_col, score_col, corr_word, p_word, corr_char, p_char, max_words, max_chars, sample_size):
    '''
    Plotly interactive visualisation
    '''
    # OPTIMIZATION: Sample data if too large for performance
    if len(data_filtered) > sample_size:
        data_sampled = data_filtered.sample(sample_size, random_state=42)
        print(f"\nNote: Sampled data to {sample_size:,} points for better Plotly performance")
        print(f"      Original filtered points: {len(data_filtered):,}")
    else:
        data_sampled = data_filtered
    
    # Create performance groups
    data_sampled = data_sampled.copy()
    data_sampled['performance_group'] = pd.cut(data_sampled[score_col], bins=3, labels=['Low', 'Medium', 'High']
    )
    
    # Create subplots
    fig = make_subplots(
        rows=1, 
        cols=3,
        subplot_titles=(
            f'Word Count vs Funny Score (r: {corr_word:.3f}, p: {p_word:.3f})',
            f'Character Count vs Funny Score (r: {corr_char:.3f}, p: {p_char:.3f})',
            'Word Count by Performance Group'
        ),
        horizontal_spacing=0.15
    )
    
    # Plot 1: Word Count Scatter
    fig.add_trace(
        go.Scatter(x=data_sampled[word_col], y=data_sampled[score_col], mode='markers',
                marker=dict(size=4, color=COLORS['primary'], opacity=0.3, line=dict(width=0)),name='Captions', showlegend=False, hovertemplate='<b>Words:</b> %{x}<br><b>Score:</b> %{y:.2f}<extra></extra>'), row=1, col=1)
    
    # Add trendline for scatter 1
    if len(data_sampled) > 1:
        try:
            z = np.polyfit(data_sampled[word_col], data_sampled[score_col], 1)
            p = np.poly1d(z)
            x_sorted = np.sort(data_sampled[word_col])
            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=p(x_sorted),
                    mode='lines',
                    line=dict(color=COLORS['accent1'], width=2),
                    name='Trend',
                    showlegend=False,
                    hovertemplate='Trend line<extra></extra>'
                ),
                row=1, col=1
            )
        except:
            pass  
    
    # Plot 2: Character Count Scatter
    fig.add_trace(
        go.Scatter(
            x=data_sampled[char_col],
            y=data_sampled[score_col],
            mode='markers',
            marker=dict(
                size=4,
                color=COLORS['accent2'],
                opacity=0.3,
                line=dict(width=0)
            ),
            name='Captions',
            showlegend=False,
            hovertemplate='<b>Chars:</b> %{x}<br><b>Score:</b> %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Add trendline for scatter 2
    if len(data_sampled) > 1:
        try:
            z = np.polyfit(data_sampled[char_col], data_sampled[score_col], 1)
            p = np.poly1d(z)
            x_sorted = np.sort(data_sampled[char_col])
            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=p(x_sorted),
                    mode='lines',
                    line=dict(color=COLORS['accent1'], width=2),
                    name='Trend',
                    showlegend=False,
                    hovertemplate='Trend line<extra></extra>'
                ),
                row=1, col=2
            )
        except:
            pass  # Skip trendline if calculation fails
    
    # Plot 3: Box plot
    groups = ['Low', 'Medium', 'High']
    colors_box = [COLORS['secondary'], COLORS['accent3'], COLORS['accent4']]
    
    for i, group in enumerate(groups):
        group_data = data_sampled[data_sampled['performance_group'] == group][word_col].dropna()
        
        if len(group_data) > 0:
            fig.add_trace(
                go.Box(
                    y=group_data,
                    name=group,
                    boxpoints=False,  # No points for better performance
                    marker_color=colors_box[i],
                    line=dict(color=colors_box[i]),
                    showlegend=False,
                    hovertemplate='<b>Group:</b> ' + group + 
                                 '<br><b>Median:</b> %{y:.1f}<br>' +
                                 '<b>Q1:</b> %{q1:.1f}<br>' +
                                 '<b>Q3:</b> %{q3:.1f}<extra></extra>'
                ),
                row=1, col=3
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Caption Performance Analysis (n={len(data_sampled):,})",
            font=dict(family=FONT_FAMILY, size=20),
            x=0.5,
            xanchor='center'
        ),
        font=dict(family=FONT_FAMILY, size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        height=500,
        margin=dict(t=100)  # Extra margin for title
    )
    
    # Update axes for plot 1
    fig.update_xaxes(
        title_text="Word Count",
        gridcolor='rgba(230, 230, 230, 0.5)',
        row=1, col=1,
        range=[0, max_words]
    )
    fig.update_yaxes(
        title_text="Funny Score",
        gridcolor='rgba(230, 230, 230, 0.5)',
        row=1, col=1,
        range=[data_sampled[score_col].min() - 0.1, data_sampled[score_col].max() + 0.1]
    )
    
    # Update axes for plot 2
    fig.update_xaxes(
        title_text="Character Count",
        gridcolor='rgba(230, 230, 230, 0.5)',
        row=1, col=2,
        range=[0, max_chars]
    )
    fig.update_yaxes(
        title_text="Funny Score",
        gridcolor='rgba(230, 230, 230, 0.5)',
        row=1, col=2,
        range=[data_sampled[score_col].min() - 0.1, data_sampled[score_col].max() + 0.1]
    )
    
    # Update axes for plot 3
    fig.update_xaxes(
        title_text="Performance Group",
        gridcolor='rgba(230, 230, 230, 0.5)',
        row=1, col=3
    )
    fig.update_yaxes(
        title_text="Word Count",
        gridcolor='rgba(230, 230, 230, 0.5)',
        row=1, col=3
    )
    
    # Update subplot titles font
    for annotation in fig.layout.annotations:
        annotation.font.size = 16
        annotation.font.family = FONT_FAMILY
    
    # Show summary statistics
    print(f"\nDisplaying {len(data_sampled):,} data points")
    print(f"Filter limits: words ≤ {max_words}, chars ≤ {max_chars}")
    print(f"Original data points: {len(data):,}")
    print(f"Filtered data points: {len(data_filtered):,}")
    
    # Performance groups summary
    print(f"\nPerformance groups distribution:")
    for group in groups:
        count = len(data_sampled[data_sampled['performance_group'] == group])
        if count > 0:
            mean_words = data_sampled[data_sampled['performance_group'] == group][word_col].mean()
            print(f"  - {group}: {count} captions (avg {word_col}: {mean_words:.1f})")
    
    # Show the plot
    fig.show()
    
    # Return figure object and statistics
    return {
        'figure': fig,
        'corr_word': corr_word,
        'p_word': p_word,
        'corr_char': corr_char,
        'p_char': p_char,
        'data_sampled': data_sampled,
        'data_filtered': data_filtered,
        'stats': {
            'total_points': len(data),
            'filtered_points': len(data_filtered),
            'displayed_points': len(data_sampled),
            'word_corr_strength': interpretation_correlation(corr_word),
            'char_corr_strength': interpretation_correlation(corr_char)
        }
    }

def create_matplotlib_visualization(data, data_filtered, word_col, char_col, score_col,
                                   corr_word, p_word, corr_char, p_char):
    """
    Create Matplotlib static visualization (original style).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Word count vs Score
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=data_filtered, x=word_col, y=score_col, alpha=0.6)
    plt.title(f'Word count vs funny score\n(r: {corr_word:.3f}, p: {p_word:.3f})')
    plt.xlabel('Word Count')
    plt.ylabel('Funny Score')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Character count vs Score
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=data_filtered, x=char_col, y=score_col, alpha=0.6)
    plt.title(f'Character count vs funny score\n(r: {corr_char:.3f}, p: {p_char:.3f})')
    plt.xlabel('Character Count')
    plt.ylabel('Funny Score')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Box plot
    plt.subplot(1, 3, 3)
    data_copy = data.copy()
    data_copy['performance_group'] = pd.cut(data_copy[score_col], bins=3, 
                                           labels=['Low', 'Medium', 'High'])
    sns.boxplot(data=data_copy, x='performance_group', y=word_col)
    plt.title('Word count by performance group')
    plt.xlabel('Performance Group')
    plt.ylabel('Word Count')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Caption Analysis (n={len(data_filtered):,})', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    return {
        'corr_word': corr_word,
        'p_word': p_word,
        'corr_char': corr_char,
        'p_char': p_char,
        'data_filtered': data_filtered
    }

def create_plotly_visualization(data, data_filtered, word_col, char_col, score_col,
                                   corr_word, p_word, corr_char, p_char):
    """
    Create Matplotlib static visualization (original style).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Word count vs Score
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=data_filtered, x=word_col, y=score_col, alpha=0.6)
    plt.title(f'Word count vs funny score\n(r: {corr_word:.3f}, p: {p_word:.3f})')
    plt.xlabel('Word Count')
    plt.ylabel('Funny Score')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Character count vs Score
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=data_filtered, x=char_col, y=score_col, alpha=0.6)
    plt.title(f'Character count vs funny score\n(r: {corr_char:.3f}, p: {p_char:.3f})')
    plt.xlabel('Character Count')
    plt.ylabel('Funny Score')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Box plot
    plt.subplot(1, 3, 3)
    data_copy = data.copy()
    data_copy['performance_group'] = pd.cut(data_copy[score_col], bins=3, 
                                           labels=['Low', 'Medium', 'High'])
    sns.boxplot(data=data_copy, x='performance_group', y=word_col)
    plt.title('Word count by performance group')
    plt.xlabel('Performance Group')
    plt.ylabel('Word Count')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Caption Analysis (n={len(data_filtered):,})', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    return {
        'corr_word': corr_word,
        'p_word': p_word,
        'corr_char': corr_char,
        'p_char': p_char,
        'data_filtered': data_filtered
    }

def print_and_plot_proportion_correlation(data, word_col='word_counts', char_col = 'length', score_col = 'success_metric', max_words = 100, max_chars = 400, use_plotly=True, sample_size = 10000): 
    '''
    Print correlation statistics and visualisations
    '''
    corr_word_score, p_word = pearsonr(data[word_col], data[score_col])
    corr_char_score, p_char = pearsonr(data[char_col], data[score_col])

    # print correlation results: 
    print("Correlation analysis: ")
    print(f"Word count correlation: {interpretation_correlation(corr_word_score)} {'positive' if corr_word_score > 0 else 'negative'} relationship")
    print(f"\nStatistical significance (α=0.05):")
    print(f"Word count: {'Significant' if p_word < 0.05 else 'Not significant'}")

    print(f"Character count correlation: {interpretation_correlation(corr_char_score)} {'positive' if corr_char_score > 0 else 'negative'} relationship")
    print(f"\nStatistical significance (α=0.05):")
    print(f"Character count: {'Significant' if p_char < 0.05 else 'Not significant'}")

    # filtered dataframe for cleaner scatter plots
    data_filtered = data[(data[word_col] <= max_words) & (data[char_col] <= max_chars)]
    
    # plotting: 
    if use_plotly:
        # Create Plotly interactive visualization
        return plot_prop_step_one(data, data_filtered, 
                                          word_col, char_col, score_col,
                                          corr_word_score, p_word,
                                          corr_char_score, p_char,
                                          max_words, max_chars, sample_size)
    else:
        # Create Matplotlib static visualization (your original code)
        return create_matplotlib_visualization(data, data_filtered,
                                              word_col, char_col, score_col,
                                              corr_word_score, p_word,
                                              corr_char_score, p_char)

def theme_wordcount_analysis_plotly(df,text_col='clean_text',theme_col='theme_label',max_words=100,min_votes=5,min_theme_size=30,lowess_frac=0.3):
    """
    Compute theme-wise word counts, correlations, and visualize with Plotly.
    """
    df = df.copy()
    if 'word_counts' not in df.columns:
        df['word_counts'] = df[text_col].str.split().str.len()

    # filter extremes
    df_theme = df[(df['word_counts'] <= max_words) & (df['votes'] >= min_votes)].copy()

    # compute per-theme correlations
    theme_corrs = []
    for theme, g in df_theme.groupby(theme_col):
        if len(g) < min_theme_size:
            continue

        # scores
        g['total_votes'] = g['not_funny'] + g['somewhat_funny'] + g['funny']
        g['total_votes'] = g['total_votes'].replace(0, 1)  # avoid division by zero

        # proportions
        g['prop_not_funny'] = g['not_funny'] / g['total_votes']
        g['prop_somewhat_funny'] = g['somewhat_funny'] / g['total_votes']
        g['prop_funny'] = g['funny'] / g['total_votes']

        # weighted average
        g['funny_score_weighted'] = (g['funny']*2 + g['somewhat_funny']) / g['total_votes']
        g['success_metric'] = g['funny_score_weighted']

        r, p = pearsonr(g['word_counts'], g['success_metric'])
        theme_corrs.append({'theme': theme, 'r_word_funny': r, 'p_value': p, 'n': len(g)})

        # assign back to main df_theme for plotting
        df_theme.loc[g.index, 'success_metric'] = g['success_metric']

    theme_corr_df = pd.DataFrame(theme_corrs).sort_values('r_word_funny').reset_index(drop=True)

    # Box and scatter
    fig_dist = px.box(df_theme,x=theme_col,y='word_counts',points=False,color_discrete_sequence=[COLORS['secondary']],title="Word count distribution per theme")

    fig_dist.add_trace(
        go.Scatter(
            x=df_theme[theme_col],
            y=df_theme['word_counts'],
            mode='markers',
            marker=dict(
                color=df_theme['success_metric'],
                colorscale='Viridis',
                size=5,
                opacity=0.6,
                showscale=True,
                colorbar=dict(title="Funny score")
            ),
            hovertemplate="<b>Theme:</b> %{x}<br><b>Words:</b> %{y}<br><b>Funny score:</b> %{marker.color:.2f}<extra></extra>",
            showlegend=False
        )
    )

    fig_dist.update_layout(
        font=dict(family=FONT_FAMILY),
        xaxis_tickangle=45,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600
    )

    # LOWESS curves
    fig_lowess = go.Figure()
    palette = list(COLORS.values())

    for i, theme in enumerate(df_theme[theme_col].unique()):
        g = df_theme[df_theme[theme_col] == theme]
        if len(g) < 40:
            continue

        lowess = sm.nonparametric.lowess(
            g['success_metric'],
            g['word_counts'],
            frac=lowess_frac
        )

        fig_lowess.add_trace(
            go.Scatter(
                x=lowess[:, 0],
                y=lowess[:, 1],
                mode='lines',
                name=theme,
                line=dict(width=2, color=palette[i % len(palette)]),
                hovertemplate="<b>Theme:</b> "+theme+"<br><b>Words:</b> %{x:.0f}<br><b>Expected score:</b> %{y:.2f}<extra></extra>"
            )
        )

    fig_lowess.update_layout(
        title="Funny score vs word count per theme (LOWESS smoothing)",
        xaxis_title="Word count",
        yaxis_title="Expected funny score",
        font=dict(family=FONT_FAMILY),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        legend=dict(x=1.02, y=1)
    )

    return theme_corr_df, fig_dist, fig_lowess








def partial_correlation_manual(df, x, y, control_variables): 
    x_control = sm.add_constant(df[control_variables])
    model_x = sm.OLS(df[x], x_control).fit()
    res_x = model_x.resid
    model_y = sm.OLS(df[y], x_control).fit()
    res_y = model_y.resid

    partial_corr, p_value = pearsonr(res_x, res_y)
    return partial_corr, p_value, res_x, res_y





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm

def theme_wordcount_analysis(
    df,
    text_col='clean_text',
    theme_col='theme_label',
    max_words=100,
    min_votes=5,
    min_theme_size=30,
    lowess_frac=0.3
):
    """
    Analyze word counts per theme, calculate correlations, and visualize using matplotlib/seaborn.

    """

    df = df.copy()

    # Ensure word counts exist
    if 'word_counts' not in df.columns:
        df['word_counts'] = df[text_col].str.split().str.len()

    # Filter extremes
    df_theme = df[(df['word_counts'] <= max_words) & (df['votes'] >= min_votes)].copy()

    # Calculate theme-wise correlations
    theme_corrs = []

    for theme, g in df_theme.groupby(theme_col):
        if len(g) < min_theme_size:
            continue

        # Scores
        g['total_votes'] = g['not_funny'] + g['somewhat_funny'] + g['funny']
        g['total_votes'] = g['total_votes'].replace(0, 1)  # avoid division by zero

        # Proportions
        g['prop_not_funny'] = g['not_funny'] / g['total_votes']
        g['prop_somewhat_funny'] = g['somewhat_funny'] / g['total_votes']
        g['prop_funny'] = g['funny'] / g['total_votes']

        # Weighted average
        g['funny_score_weighted'] = (g['funny'] * 2 + g['somewhat_funny']) / g['total_votes']
        g['success_metric'] = g['funny_score_weighted']

        r, p = pearsonr(g['word_counts'], g['success_metric'])
        theme_corrs.append({
            'theme': theme,
            'r_word_funny': r,
            'p_value': p,
            'n': len(g)
        })

        # Assign success_metric back for plotting
        df_theme.loc[g.index, 'success_metric'] = g['success_metric']

    theme_corr_df = pd.DataFrame(theme_corrs).sort_values('r_word_funny').reset_index(drop=True)
    print(theme_corr_df.head(10))


    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=df_theme,
        x=theme_col,
        y='word_counts',
        color='lightgray',
        showfliers=False
    )
    sns.stripplot(
        data=df_theme,
        x=theme_col,
        y='word_counts',
        hue='success_metric',
        palette='viridis',
        alpha=0.6,
        jitter=0.25,
        size=4
    )
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Word count")
    plt.xlabel("Theme")
    plt.title("Word count distribution per theme\nColored by funny score")
    plt.legend(title="Funny score", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 8))
    themes = df_theme[theme_col].unique()
    for theme in themes:
        g = df_theme[df_theme[theme_col] == theme]
        if len(g) < 40:
            continue
        lowess = sm.nonparametric.lowess(g['success_metric'], g['word_counts'], frac=lowess_frac)
        plt.plot(lowess[:, 0], lowess[:, 1], label=theme)

    plt.xlabel("Word count")
    plt.ylabel("Expected funny score")
    plt.title("Funny score vs word count per theme (LOWESS smoothing)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.25),ncol=6,fontsize=9,frameon=False)
    #plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.tight_layout()
    plt.show()

    return theme_corr_df, df_theme




