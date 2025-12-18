import os
from pathlib import Path
from math import floor

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib.image import imread

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
import json
from datetime import datetime, timedelta
import nbimporter
from scipy.ndimage import gaussian_filter1d


#Used for the Website quizz

def computing_deltas(folder_path):
    top2_delta = {}  # delta between top 2 captions

    for contest_num in range(510, 896):
        filename = f"{contest_num}_cleaned.csv"
        file_path = os.path.join(folder_path, filename)

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if len(df) >= 2:
                    top_scores = df['mean'].nlargest(2).values
                    margin = top_scores[0] - top_scores[1]
                    top2_delta[contest_num] = margin
                else:
                    top2_delta[contest_num] = None
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                top2_delta[contest_num] = None
        else:
            top2_delta[contest_num] = None

    return top2_delta


def print_top3_biggest_deltas(folder_path):
    delta_dict = computing_deltas(folder_path)

    # Convert to DataFrame
    delta_df = pd.DataFrame(
        list(delta_dict.items()),
        columns=["Contest", "Top2_Delta"]
    ).dropna()

    # Sort by delta descending
    top3 = delta_df.sort_values("Top2_Delta", ascending=False).head(3)

    print("\nTop 3 contests with the biggest gap between top 1 and top 2 captions:\n")
    for _, row in top3.iterrows():
        print(f"Contest {int(row['Contest'])} → Delta = {row['Top2_Delta']:.3f}")


def print_top3_captions(contest_num, folder_path):
    filename = f"{contest_num}_cleaned.csv"
    file_path = os.path.join(folder_path, filename)

    if not os.path.exists(file_path):
        print(f"File not found: {filename}")
        return

    df = pd.read_csv(file_path)

    if len(df) < 3:
        print(f"Contest {contest_num} has fewer than 3 captions.")
        return

    top3 = df.nlargest(3, "mean")

    print(f"\nContest {contest_num} — Top 3 Captions by Mean Score:\n")
    for rank, (_, row) in enumerate(top3.iterrows(), start=1):
        print(
            f"Rank {rank}: "
            f"Mean = {row['mean']:.3f} | "
            f"'{row['caption']}'"
        )

### Timing analysis
def add_contest_dates(df):
    """
    Adds publish and vote dates for each contest based on weekly frequency.
    Contest 510 is assumed to be published on 22 Feb 2016.
    """

    base_publish = datetime.strptime('2016.02.22', "%Y.%m.%d")

    publish_dates = []
    vote_dates = []

    for i in range(len(df)):
        contest_id = df['contest_id'].iloc[i]
        diff = int(contest_id) - 510

        publish_date = base_publish + timedelta(days=diff * 7)
        vote_date = publish_date + timedelta(days=7)

        # Ensure metadata is a dictionary
        if df.at[i, 'metadata'] is None:
            df.at[i, 'metadata'] = {}

        # Now safe to assign
        df.at[i, 'metadata']['contest_day_publish'] = publish_date
        df.at[i, 'metadata']['contest_day_votes'] = vote_date

        publish_dates.append(publish_date)
        vote_dates.append(vote_date)

    df['contest_day_publish'] = publish_dates
    df['contest_day_votes'] = vote_dates

    return df

def plot_captions_and_funniness_timeline_plotly(
    folder_path,
    contest_with_dates,
    sigma=10
):
    folder_path = str(folder_path)
    records = []

    # --- Loop over contests ---
    for contest_num in range(510, 896):
        filename = f"{contest_num}_cleaned.csv"
        file_path = os.path.join(folder_path, filename)

        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path)

            records.append({
                "Contest": contest_num,
                "Num_Captions": len(df),
                "Mean_Funniness": df["mean"].mean()
            })

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # --- Create DataFrame ---
    df_metrics = pd.DataFrame(records)

    # --- Merge with publish dates ---
    merged = (
        df_metrics
        .merge(
            contest_with_dates[["contest_id", "contest_day_publish"]],
            left_on="Contest",
            right_on="contest_id",
            how="inner"
        )
        .sort_values("contest_day_publish")
    )

    # --- Gaussian smoothing ---
    merged["smooth_captions"] = gaussian_filter1d(
        merged["Num_Captions"], sigma=sigma
    )
    merged["smooth_funniness"] = gaussian_filter1d(
        merged["Mean_Funniness"], sigma=sigma
    )

    # --- Build Plotly figure ---
    fig = go.Figure()

    # ===== Y1: Number of captions =====
    fig.add_trace(go.Scatter(
        x=merged["contest_day_publish"],
        y=merged["Num_Captions"],
        mode="markers",
        name="Captions (raw)",
        opacity=0.5,
        marker=dict(size=6),
        yaxis="y1",
        customdata=merged["Contest"],
        hovertemplate=(
            "<b>Contest</b>: %{customdata}<br>"
            "<b>Number of Captions</b>: %{y}"
            "<extra></extra>"
        )

        
    ))

    fig.add_trace(go.Scatter(
        x=merged["contest_day_publish"],
        y=merged["smooth_captions"],
        mode="lines",
        line=dict(color="darkgoldenrod", width=3),
        name=f"Captions (smooth) (σ={sigma})",
        yaxis="y1",
        hovertemplate=(
            "<b>Smoothed number of captions</b>: %{y:.2f}<extra></extra>"
        )
    ))

    # ===== Y2: Mean funniness =====
    fig.add_trace(go.Scatter(
        x=merged["contest_day_publish"],
        y=merged["Mean_Funniness"],
        mode="markers",
        name="Mean funniness (raw)",
        opacity=0.5,
        marker=dict(size=6, color="firebrick"),
        yaxis="y2",
        hovertemplate=(
            "<b>Mean funniness</b>: %{y:.2f}<extra></extra>"
        )
    ))

    fig.add_trace(go.Scatter(
        x=merged["contest_day_publish"],
        y=merged["smooth_funniness"],
        mode="lines",
        line=dict(color="firebrick", width=3),
        name=f"Mean funniness (smooth) (σ={sigma})",
        yaxis="y2",
        hovertemplate=(
            "<b>Smoothed mean funniness</b>: %{y:.2f}<extra></extra>"
        )
    ))

    # --- Layout ---
    fig.update_layout(
        title="Contest Popularity and Mean Funniness Over Time",
        xaxis=dict(
            title="Publish Date",
            rangeslider=dict(visible=False)  # ← THIS LINE
        ),
        yaxis=dict(
            title="Number of Captions",
            side="left"
        ),
        yaxis2=dict(
            title="Mean Funniness",
            overlaying="y",
            side="right",
            rangemode="normal"
        ),
        template="for_dummies",
        hovermode="x unified",
        width=740,
        height=600
    )
    # --- Save and show ---
    save_path = (
        r"C:\Users\hanna\ADA\ada-2025-project-json-derulo"
        r"\docs\plots\captions_funniness_timeline_plot.html"
    )

    fig.write_html(save_path, include_plotlyjs="cdn")
    fig.show()

    return merged