
import pandas as pd
import json
from pathlib import Path

from tqdm.auto import tqdm
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import silhouette_score
from IPython.display import HTML
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.nonparametric.smoothers_lowess import lowess



def compute_theme_term_sizes(df_tfidf, df_themes, multiplier='nb_instances'):
    """
    Minimal: returns DataFrame with columns ['term','theme_label','Size'].
    Size = (score / sum(scores_in_theme)) * multiplier
    """
    # base merge to bring theme label and nb_instances
    df = df_tfidf.merge(df_themes[['theme_id', 'theme_label', multiplier]],
                        on='theme_id', how='left')

    # sum of scores per theme (protect against missing/zero sums)
    total = df.groupby('theme_id')['score'].transform('sum')
    

    # compute Size
    df['Size'] = df['score'] / total * df[multiplier]

    return df[['term', 'theme_label', 'Size']].reset_index(drop=True)


def compute_theme_term_sizes_from_merged(df, label_col="theme_label_llm", multiplier="nb_instances"):
    total = df.groupby("theme_id")["score"].transform("sum")
    df = df.copy()
    df["Size"] = df["score"] / total * df[multiplier]
    return df[["term", label_col, "Size"]].rename(columns={label_col: "theme_label"}).reset_index(drop=True)





def generate_olama_themes(themes_df, client):
    results = {}

    for id, row in tqdm(themes_df.iterrows()):
        tf_idf = row["TF_IDF_terms"]

        prompt = (
            f"Here are the TF-IDF terms for theme {id} : {tf_idf}. "
            f"Please generate a word representative of the 10 tf-idf terms \n"
        )
        prompt += "please your answer must be as following : id=... , LLMS_theme=..."

        messages = [{
            "role": "user",
            "content": prompt,
        }]

        full_response = ""
        for part in client.chat("gpt-oss:120b", messages=messages, stream=True):
            if part.message.content:
                full_response += part.message.content

        results[str(id)] = full_response

    rows = []
    for cap_id, text in results.items():
        parts = [p.strip() for p in text.split(",")][1]
        label = parts.removeprefix("LLMS_theme=")

        row = {
            "caption_id": cap_id,
            "gen_label": label
        }
        rows.append(row)

    return pd.DataFrame(rows)



def merge_llm_theme_labels(tfidf_df, themes_df, llm_df):
    """
    Replace theme labels with LLM-generated labels when available
    and merge them into the TF-IDF dataframe.

    Parameters
    ----------
    tfidf_df : pd.DataFrame
        TF-IDF dataframe containing 'theme_id'.
    themes_df : pd.DataFrame
        Theme metadata with columns ['theme_id', 'theme_label', 'nb_instances'].
    llm_df : pd.DataFrame
        LLM output with columns ['caption_id', 'gen_label'].

    Returns
    -------
    pd.DataFrame
        TF-IDF dataframe enriched with:
        - theme_label_llm
        - nb_instances
    """
    # Build mapping: theme_id -> LLM label
    theme_id_to_llm = dict(
        zip(llm_df["caption_id"].astype(int), llm_df["gen_label"])
    )

    # Add LLM label column (fallback to original label)
    themes_df = themes_df.copy()
    themes_df["theme_label_llm"] = (
        themes_df["theme_id"]
        .map(theme_id_to_llm)
        .fillna(themes_df["theme_label"])
        .str.title()
    )

    # Merge into TF-IDF dataframe
    df = tfidf_df.merge(
        themes_df[["theme_id", "theme_label_llm", "nb_instances"]],
        on="theme_id",
        how="left"
    )

    return df



def plot_funiness_vs_frequency(themes_df):
    """
    Scatter plot of theme frequency vs mean funniness with a LOWESS trendline.

    Parameters
    ----------
    themes_df : pd.DataFrame
        Must contain columns:
        - nb_instances (theme frequency)
        - mu_mean (mean funniness score)
        - theme_id
        - llm_theme_label

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = px.scatter(
        themes_df,
        x="nb_instances",
        y="mu_mean",
        hover_data=["theme_id", "llm_theme_label"],
        labels={
            "nb_instances": "Number of captions (frequency)",
            "mu_mean": "Mean of captions (μ)",
            "llm_theme_label": "Caption theme",
        },
        title="Themes funniness vs frequency with LOWESS trendline",
        color="llm_theme_label",
    )

    fig.update_traces(
        marker=dict(
            size=10,
            opacity=0.7,
            line=dict(width=1, color="black")
        )
    )

    lowess_line = lowess(
        endog=themes_df["mu_mean"],
        exog=themes_df["nb_instances"],
        frac=0.25,
        it=0,
        return_sorted=True,
    )

    fig.add_trace(
        go.Scatter(
            x=lowess_line[:, 0],
            y=lowess_line[:, 1],
            mode="lines",
            line=dict(color="crimson", width=3),
            name="LOWESS trend",
            hoverinfo="skip",
        )
    )

    return fig



def compute_yearly_theme_funniness_shifted(docs_with_themes):
    """
    Compute yearly mean funniness per theme, normalize by yearly baseline,
    and shift values to be strictly positive.

    Returns a DataFrame with:
    - contest_publish_year
    - theme_label_llm
    - mean_funniness
    - year_mean
    - funniness_delta
    - funniness_shifted
    """
    # Yearly mean funniness per theme
    yearly_theme_funniness = (
        docs_with_themes
        .groupby(["contest_publish_year", "theme_label_llm"])["mean"]
        .mean()
        .reset_index(name="mean_funniness")
    )

    # Yearly baseline funniness
    yearly_theme_funniness["year_mean"] = (
        yearly_theme_funniness
        .groupby("contest_publish_year")["mean_funniness"]
        .transform("mean")
    )

    # Delta from yearly mean
    yearly_theme_funniness["funniness_delta"] = (
        yearly_theme_funniness["mean_funniness"]
        - yearly_theme_funniness["year_mean"]
    )

    # Shift to ensure positivity
    min_delta = yearly_theme_funniness["funniness_delta"].min()
    yearly_theme_funniness["funniness_shifted"] = (
        yearly_theme_funniness["funniness_delta"] - min_delta + 0.001
    )

    return yearly_theme_funniness




def prepare_docs_with_themes(
    docs_with_themes,
    llm_df,
    base_publish_str="2016.02.22",
    first_contest_id=510,
):
    """
    Prepare docs_with_themes by:
    - computing contest publish dates and years
    - adding LLM-generated theme labels

    Parameters
    ----------
    docs_with_themes : pd.DataFrame
        Must contain columns ['contest_id', 'theme_id'].
    llm_df : pd.DataFrame
        Must contain columns ['caption_id', 'gen_label'].
    base_publish_str : str
        Publish date of the first contest (YYYY.MM.DD).
    first_contest_id : int
        Contest ID corresponding to base_publish_str.

    Returns
    -------
    pd.DataFrame
        Enriched docs_with_themes with:
        - contest_day_publish
        - contest_publish_year
        - theme_label_llm
    """
    df = docs_with_themes.copy()

    # Ensure numeric IDs
    df["contest_id"] = df["contest_id"].astype(int)
    df["theme_id"] = df["theme_id"].astype(int)

    # Base publish date
    base_publish = datetime.strptime(base_publish_str, "%Y.%m.%d")

    # Compute publish date from weekly schedule
    df["contest_day_publish"] = df["contest_id"].apply(
        lambda cid: base_publish + timedelta(days=(cid - first_contest_id) * 7)
    )

    # Extract year
    df["contest_publish_year"] = df["contest_day_publish"].dt.year

    # Build mapping theme_id -> LLM label
    llm_map = dict(
        zip(llm_df["caption_id"].astype(int), llm_df["gen_label"])
    )

    # Add LLM theme label
    df["theme_label_llm"] = (
        df["theme_id"]
        .map(llm_map)
        .fillna("Unknown")
        .str.title()
    )

    return df

