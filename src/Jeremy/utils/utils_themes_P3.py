# This file holds the utiliy functions concerning the themes extraction of the newyorker caption contest dataset.

# general
from pathlib import Path

# Data handling
import pandas as pd
import numpy as np

# plot
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ================================================== #
# =================== DATA LOADING ================= #
# ================================================== #

def load_contests(contests_json_path: Path):
    with open(contests_json_path, "r", encoding="utf-8") as f:
        contests = json.load(f)
    return contests

def build_docs_dataframe(csv_path, json_path, debug=False):
    contests = load_contests(json_path)
    docs = []
    error_csv = [] # to deal with badly formatted csvs

    for contest in tqdm(contests, desc="Step 1: data loading/building"):
        cid = contest["contest_id"]
        raw_meta = contest.get("metadata")
        meta = raw_meta or {}

        # rassembler "image document" par contest
        parts = []
        for key in ["image_descriptions", "image_uncanny_descriptions",
                    "image_locations", "questions"]:
            values = meta.get(key) or []
            parts.extend(values)

        entities = meta.get("entities") or []
        for url in entities:
            last = str(url).rstrip("/").split("/")[-1]  # get last part of URL
            parts.append(last.replace("_", " "))        #erase understcores

        image_text = " ".join(map(str, parts)).strip()  # concat all parts

        # add if non empty (a lot of contests have no image metadata near the end of the dataset)
        if image_text:
            docs.append(
                {
                    "doc_id": f"image_{cid}",
                    "contest_id": cid,
                    "kind": "image",
                    "text": image_text,
                }
            )

        # add caption documents (still per contest)
        csv_path = csv_path / f"{cid}.csv"
        if not csv_path.exists(): # in case using the cleaned csv files
            csv_path = csv_path / f"{cid}_cleaned.csv"
            if not csv_path.exists():
                print(f"Warning: CSV for contest {cid} not found at {csv_path / f'{cid}.csv'} and neither at {csv_path}.")
                continue

        try:
            captions_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading CSV for contest {cid} at {csv_path}: {e}")
            error_csv.append(cid)
            continue

        # optional inner tqdm if you want per-caption progress too
        for idx, row in captions_df.iterrows():
            cap_text = str(row.get("caption", "")).strip()
            if not cap_text:
                continue

            docs.append(
                {
                    "doc_id": f"caption_{cid}_{idx}",
                    "contest_id": cid,
                    "kind": "caption",
                    "text": cap_text,
                    "rank": row.get("rank", None),
                    "mean": row.get("mean", None),
                    "precision": row.get("precision", None),
                    "votes": row.get("votes", None),
                    "not_funny": row.get("not_funny", None),
                    "somewhat_funny": row.get("somewhat_funny", None),
                    "funny": row.get("funny", None),
                }
            )

    df = pd.DataFrame(docs)
    
    # keep only expected first 1000 rows for debugging
    if debug:
        df = df.head(5000)
    
    return df, error_csv

# other loading function to load from an output csv that comes from this notebook
def load_docs_with_clean_text(docs_csv_path, debug=False):
    """
    Load the prebuilt docs_with_clean_text.csv file so you don’t have to rebuild
    docs from individual contest files. Returns the dataframe plus an empty
    error list for API parity with build_docs_dataframe.
    """
    docs_csv_path = Path(docs_csv_path)
    if not docs_csv_path.exists():
        raise FileNotFoundError(f"docs_with_clean_text.csv not found at {docs_csv_path}")

    cols = [
        "doc_id", "contest_id", "kind", "text",
        "rank", "mean", "precision", "votes",
        "not_funny", "somewhat_funny", "funny",
        "clean_text",
    ]
    df = pd.read_csv(docs_csv_path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"docs_with_clean_text.csv is missing expected columns: {missing}")
    
    # remove rows with NAN (the embedding step cannot handle those)
    # df["text"] = df["text"].fillna("")
    
    # keep only expected first 1000 rows for debugging
    if debug:
        df = df.head(5000)

    return df, []


# ================================================================================================ #
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
# ================================================================================================ #



# ================================================== #
# ===================== PLOTTING =================== #
# ================================================== #

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def heatmap_from_df(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    image_theme_col: str = "image_theme_id",
    colors=None,
    title: str = "Heatmap",
    width: int = 5000,          # total figure width in pixels
    colorbar_px: int = 90,      # approx width reserved for colorbar
    pad_px: int = 24,           # small padding/fudge factor
    extra_top_px: int = 80,     # title/top padding
    min_cell_px: int = 3,       # clamp cell size so it doesn't get too tiny
    max_cell_px: int = 30,      # clamp cell size so it doesn't get huge
    categorical_axes: str = "auto",  # "auto" | True | False
):
    """
    Builds a 2x2 subplot:
      - main heatmap (row1,col1)
      - right-side average strip over y (row1,col2)
      - bottom average strip over x (row2,col1)

    Key feature: if your x/y ids are sparse (e.g. 1,2,5,8,74), it will plot them
    as *categorical* axes (no gaps), while keeping square cells via sizing + scaleanchor.
    """

    # ---- Validate duplicates ----
    dup_mask = df.duplicated(subset=[x_col, y_col], keep=False)
    if dup_mask.any():
        dup_pairs = df.loc[dup_mask, [x_col, y_col]].drop_duplicates()
        raise ValueError(
            "Duplicate (x, y) pairs found; provide an aggregation or clean data. Examples:\n"
            f"{dup_pairs.head()}"
        )

    # ---- Colorscale ----
    if colors is None:
        colors = px.colors.sequential.Viridis
    if len(colors) < 2:
        raise ValueError("Provide at least two colors to build a gradient.")
    stops = [i / (len(colors) - 1) for i in range(len(colors))]
    colorscale = list(zip(stops, colors))

    # ---- Pivot + averages ----
    pivot = df.pivot(index=y_col, columns=x_col, values=value_col)

    theme_means = df.groupby(image_theme_col)[value_col].mean()
    caption_means = df.groupby(x_col)[value_col].mean()

    # ---- Decide whether to treat axes as categorical (no gaps) ----
    def _is_sparse_numeric(vals: np.ndarray) -> bool:
        vals = np.asarray(vals)
        if vals.size < 2:
            return False
        # consecutive integers => not sparse
        v = np.unique(vals.astype(float))
        v = v[np.isfinite(v)]
        if v.size < 2:
            return False
        # if theme IDs are integers, "consecutive" means max-min+1 == count
        if np.all(np.isclose(v, np.round(v))):
            return (int(np.nanmax(v)) - int(np.nanmin(v)) + 1) != int(v.size)
        # otherwise treat as sparse
        return True

    x_is_num = np.issubdtype(pivot.columns.dtype, np.number)
    y_is_num = np.issubdtype(pivot.index.dtype, np.number)

    if categorical_axes == "auto":
        use_cat_x = (not x_is_num) or _is_sparse_numeric(pivot.columns.values)
        use_cat_y = (not y_is_num) or _is_sparse_numeric(pivot.index.values)
    else:
        use_cat_x = bool(categorical_axes)
        use_cat_y = bool(categorical_axes)

    # ---- Order axes and (if categorical) create label arrays without gaps ----
    # We always reorder pivot so the displayed order is stable.
    if x_is_num:
        x_sorted = sorted(pivot.columns.values.tolist(), key=lambda v: float(v))
    else:
        x_sorted = sorted(pivot.columns.values.tolist(), key=lambda v: str(v))

    if y_is_num:
        y_sorted = sorted(pivot.index.values.tolist(), key=lambda v: float(v))
    else:
        y_sorted = sorted(pivot.index.values.tolist(), key=lambda v: str(v))

    pivot = pivot.reindex(index=y_sorted, columns=x_sorted)

    # labels used in the plot (categories must be strings to avoid Plotly treating them as continuous)
    x_labels = [str(v) for v in x_sorted] if use_cat_x else list(pivot.columns)
    y_labels = [str(v) for v in y_sorted] if use_cat_y else list(pivot.index)

    # Averages aligned to the reordered axes
    theme_means_aligned = theme_means.reindex(y_sorted).values.reshape(-1, 1)
    caption_means_aligned = caption_means.reindex(x_sorted).values.reshape(1, -1)

    # ---- Subplot geometry ----
    col_widths = [0.93, 0.07]
    row_heights = [0.99, 0.15]

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_yaxes=False,
        shared_xaxes=False,
        column_widths=col_widths,
        row_heights=row_heights,
        horizontal_spacing=0.005,
        vertical_spacing=0.05,
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "domain"}],
        ],
        subplot_titles=(title, "Average", None, None),
    )

    # ---- Traces ----
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=x_labels,
            y=y_labels,
            coloraxis="coloraxis",
            hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{value_col}: %{{z}}<extra></extra>",
            showscale=True,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=theme_means_aligned,
            x=["avg"],
            y=y_labels,
            coloraxis="coloraxis",
            showscale=False,
            hovertemplate=f"{image_theme_col}: %{{y}}<br>mean {value_col}: %{{z}}<extra></extra>",
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Heatmap(
            z=caption_means_aligned,
            x=x_labels,
            y=["avg"],
            coloraxis="coloraxis",
            showscale=False,
            hovertemplate=f"{x_col}: %{{x}}<br>mean {value_col}: %{{z}}<extra></extra>",
        ),
        row=2, col=1
    )

    # ---- Layout basics ----
    fig.update_layout(
        title=None,
        coloraxis=dict(colorscale=colorscale, colorbar_title=value_col),
        plot_bgcolor="white",
        margin=dict(l=12, r=12, t=50, b=12),
        xaxis_title=None,
        yaxis_title=y_col,
        xaxis2=dict(visible=False),
        yaxis2=dict(title=None, showticklabels=False),
        xaxis3_title=x_col,
        yaxis3=dict(title="Average", showticklabels=False),
    )

    # ---- Axis alignment ----
    fig.update_yaxes(matches="y", row=1, col=2, showticklabels=False)
    fig.update_xaxes(matches="x", row=2, col=1, showticklabels=False)

    # ---- If categorical, force category ordering to match our sorted labels ----
    # Main heatmap axes are xaxis/yaxis. Bottom heatmap xaxis3 matches xaxis.
    if use_cat_x:
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=x_labels,
            row=1, col=1,
        )
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=x_labels,
            row=2, col=1,
        )

    if use_cat_y:
        fig.update_yaxes(
            type="category",
            categoryorder="array",
            categoryarray=y_labels,
            row=1, col=1,
        )
        fig.update_yaxes(
            type="category",
            categoryorder="array",
            categoryarray=y_labels,
            row=1, col=2,  # matched y; keep ordering consistent
        )

    # ---- Size: choose total width; cell size adapts ----
    n_rows, n_cols = pivot.shape

    # width ≈ main_w + right_strip_w + colorbar + padding
    # main_w = n_cols * cell_px
    # right_strip_w = main_w * (col_widths[1]/col_widths[0])
    k = 1.0 + (col_widths[1] / col_widths[0])
    usable_w = max(1, width - colorbar_px - pad_px)
    cell_px = usable_w / (n_cols * k)
    cell_px = int(np.clip(cell_px, min_cell_px, max_cell_px))

    main_w = int(n_cols * cell_px)
    main_h = int(n_rows * cell_px)
    right_strip_w = int(main_w * (col_widths[1] / col_widths[0]))
    bottom_strip_h = int(main_h * (row_heights[1] / row_heights[0]))

    fig.update_layout(
        width=main_w + right_strip_w + colorbar_px + pad_px,
        height=main_h + bottom_strip_h + extra_top_px,
    )

    # ---- Enforce square cells (works for both numeric and categorical axes) ----
    # With the sizing above, Plotly typically won’t need to pad ranges.
    fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="range", row=1, col=1)
    fig.update_xaxes(constrain="range", row=1, col=1)

    return fig
