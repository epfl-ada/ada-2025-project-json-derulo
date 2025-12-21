# This file holds the utiliy functions concerning the themes extraction of the newyorker caption contest dataset.

# general
from pathlib import Path
from tqdm import tqdm

# Data handling
import json
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

def load_csv(csv_path: Path, expected_cols) -> pd.DataFrame:
    """Load a csv file into Pandas DataFrame and check for expected columns."""
    
    df = pd.read_csv(csv_path)
    
    # check columns
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing expected columns: {missing}")
    
    return df

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
# ================== DATA FORMATTING =============== #
# ================================================== #

def build_flourish_csv(df):

    # --- Build Flourish heatmap links with labels + explicit ids ---
    # Output columns: ["Image_Theme", "Caption_Theme", "count", "funny_score", "image_theme_id", "caption_theme_id"]

    # Keep only needed columns
    cols = ["contest_id", "kind", "mean", "theme_id", "gen_label"]  # "mean" is the average funniness score of the caption/image
    df = df.loc[:, cols].copy()

    # Light memory cleanup
    df["contest_id"] = pd.to_numeric(df["contest_id"], errors="coerce").astype("Int32")
    df["theme_id"]   = pd.to_numeric(df["theme_id"], errors="coerce").astype("Int16")
    df["gen_label"]  = df["gen_label"].astype("string")

    # Theme id -> label mapping (for Source/Target strings)
    theme_map = (
        df[["theme_id", "gen_label"]]
        .dropna(subset=["theme_id", "gen_label"])
        .drop_duplicates(subset=["theme_id"])
        .set_index("theme_id")["gen_label"]
        .to_dict()
    )

    # All theme IDs (aim to include all 80)
    all_ids = sorted([int(x) for x in theme_map.keys() if pd.notna(x)])
    if len(all_ids) != 80 and all_ids and min(all_ids) >= 1 and max(all_ids) <= 80:
        all_ids = list(range(1, 81))

    # Split: one image per contest
    df_img = (
        df[df["kind"].eq("image")]
        .dropna(subset=["contest_id", "theme_id"])
        .drop_duplicates(subset=["contest_id"])
        .rename(columns={"theme_id": "img_theme_id"})
        [["contest_id", "img_theme_id"]]
    )

    # Captions (keep mean so we can aggregate funny_score per connection)
    df_cap = (
        df[df["kind"].eq("caption")]
        .dropna(subset=["contest_id", "theme_id"])
        .rename(columns={"theme_id": "cap_theme_id", "mean": "cap_mean"})
        [["contest_id", "cap_theme_id", "cap_mean"]]
    )

    # Pair captions with their contest's image theme
    pairs = df_cap.merge(df_img, on="contest_id", how="inner", validate="many_to_one")

    # Aggregate per connection: count + avg caption funniness (funny_score)
    links = (
        pairs.groupby(["img_theme_id", "cap_theme_id"], sort=False)
        .agg(
            count=("contest_id", "size"),
            funny_score=("cap_mean", "mean"),
        )
        .reset_index()
        .rename(columns={"img_theme_id": "image_theme_id", "cap_theme_id": "caption_theme_id"})
    )

    # Build full 80x80 grid so Flourish can show all themes (missing pairs -> 0)
    full_grid = pd.MultiIndex.from_product(
        [all_ids, all_ids],
        names=["image_theme_id", "caption_theme_id"]
    ).to_frame(index=False)

    df_flourish = full_grid.merge(links, on=["image_theme_id", "caption_theme_id"], how="left")

    # Fill missing values for absent connections
    df_flourish["count"] = df_flourish["count"].fillna(0).astype(int)

    # If a pair never occurs, keep funny_score as NaN (often better than forcing 0);
    # then center using the mean over observed connections only.
    # df_flourish["funny_score"] = df_flourish["funny_score"] - df_flourish["funny_score"].mean(skipna=True)

    # Build Source/Target labels like "T25: Manhattan" while keeping ids as separate cols
    def _label(tid: int) -> str:
        lab = theme_map.get(int(tid), "")
        return f"T{int(tid)}: {lab}" if lab else f"T{int(tid)}"

    df_flourish["Image_Theme"] = df_flourish["image_theme_id"].apply(_label)
    df_flourish["Caption_Theme"] = df_flourish["caption_theme_id"].apply(_label)

    # Final column order for Flourish
    df_flourish = df_flourish[
        ["Image_Theme", "Caption_Theme", "count", "funny_score", "image_theme_id", "caption_theme_id"]
    ]

    return df_flourish



def build_theme_id_matrix(df: pd.DataFrame, matrix_value: str) -> pd.DataFrame:
    """
    Build an (image_theme_id x caption_theme_id) matrix as a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe containing columns:
        ["contest_id", "kind", "mean", "theme_id", "gen_label"]
        - "kind" has values like "image" and "caption"
        - "mean" is the average funniness score (used for funny_score aggregation)
    matrix_value : str
        Either "count" or "funny_score".
        - "count": number of caption-image theme connections
        - "funny_score": mean of caption "mean" per connection

    Returns
    -------
    pd.DataFrame
        A matrix with image theme_id as index (rows) and caption theme_id as columns.
        Values are either counts or funny_score depending on matrix_value.
        - For "count": missing pairs are filled with 0 (int).
        - For "funny_score": missing pairs remain NaN (float).
    """
    if matrix_value not in {"count", "funny_score"}:
        raise ValueError("matrix_value must be either 'count' or 'funny_score'")

    needed = ["contest_id", "kind", "mean", "theme_id", "gen_label"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    # Keep only needed columns
    df = df.loc[:, needed].copy()

    # Light memory cleanup / typing
    df["contest_id"] = pd.to_numeric(df["contest_id"], errors="coerce").astype("Int32")
    df["theme_id"]   = pd.to_numeric(df["theme_id"], errors="coerce").astype("Int16")
    df["gen_label"]  = df["gen_label"].astype("string")

    # Theme id -> label mapping (to infer full set of theme ids)
    theme_map = (
        df[["theme_id", "gen_label"]]
        .dropna(subset=["theme_id", "gen_label"])
        .drop_duplicates(subset=["theme_id"])
        .set_index("theme_id")["gen_label"]
        .to_dict()
    )

    # Try to include all theme IDs (commonly 1..80); otherwise use observed ids
    all_ids = sorted([int(x) for x in theme_map.keys() if pd.notna(x)])
    if len(all_ids) != 80 and all_ids and min(all_ids) >= 1 and max(all_ids) <= 80:
        all_ids = list(range(1, 81))

    # One image theme per contest
    df_img = (
        df[df["kind"].eq("image")]
        .dropna(subset=["contest_id", "theme_id"])
        .drop_duplicates(subset=["contest_id"])
        .rename(columns={"theme_id": "img_theme_id"})
        [["contest_id", "img_theme_id"]]
    )

    # Captions (keep mean so we can aggregate funny_score per connection)
    df_cap = (
        df[df["kind"].eq("caption")]
        .dropna(subset=["contest_id", "theme_id"])
        .rename(columns={"theme_id": "cap_theme_id", "mean": "cap_mean"})
        [["contest_id", "cap_theme_id", "cap_mean"]]
    )

    # Pair each caption with its contest's image theme
    pairs = df_cap.merge(df_img, on="contest_id", how="inner", validate="many_to_one")

    # Aggregate per (img_theme_id, cap_theme_id)
    links = (
        pairs.groupby(["img_theme_id", "cap_theme_id"], sort=False)
        .agg(
            count=("contest_id", "size"),
            funny_score=("cap_mean", "mean"),
        )
        .reset_index()
    )

    # Full grid so the matrix includes all themes (e.g. full 80x80)
    full_grid = pd.MultiIndex.from_product(
        [all_ids, all_ids],
        names=["img_theme_id", "cap_theme_id"],
    )

    links = links.set_index(["img_theme_id", "cap_theme_id"]).reindex(full_grid)

    # Convert to matrix form
    mat = links[matrix_value].unstack("cap_theme_id")

    # Fill behavior: counts -> 0; funny_score -> keep NaN
    if matrix_value == "count":
        mat = mat.fillna(0).astype(int)

    # Nice axis names
    mat.index.name = "image_theme_id"
    mat.columns.name = "caption_theme_id"

    return mat


def export_flourish_theme_similarity(
    sim_matrix_df: pd.DataFrame,
    output_csv_path: str | Path | None = None,
    keep: str = "upper",          # "upper" (unique pairs), "lower", or "all"
    include_diagonal: bool = False,
    float_format: str = "%.6f",
) -> pd.DataFrame:
    """
    Convert a square similarity matrix into Flourish edge-list CSV:
    theme_id_S, theme_id_T, cos_similarity

    Works even if index are ints and columns are strings (or other mixed types).
    """
    df = sim_matrix_df.copy()

    # Build a stable order mapping from the matrix index order
    order = {label: i for i, label in enumerate(df.index)}

    # Long form
    long_df = (
        df.stack()
          .rename("cos_similarity")
          .reset_index()
          .rename(columns={"level_0": "theme_id_S", "level_1": "theme_id_T"})
    )

    # Map labels to positions for comparisons (no int/str issues)
    long_df["_pos_S"] = long_df["theme_id_S"].map(order)
    long_df["_pos_T"] = long_df["theme_id_T"].map(order)

    # If your columns labels differ only by type (e.g., "0" vs 0), try fallback conversion
    if long_df["_pos_T"].isna().any():
        # Try mapping stringified labels
        order_str = {str(k): v for k, v in order.items()}
        long_df.loc[long_df["_pos_T"].isna(), "_pos_T"] = (
            long_df.loc[long_df["_pos_T"].isna(), "theme_id_T"].astype(str).map(order_str)
        )
    if long_df["_pos_S"].isna().any():
        order_str = {str(k): v for k, v in order.items()}
        long_df.loc[long_df["_pos_S"].isna(), "_pos_S"] = (
            long_df.loc[long_df["_pos_S"].isna(), "theme_id_S"].astype(str).map(order_str)
        )

    # Drop any pairs we still can't map (shouldn't happen if matrix is well-formed)
    long_df = long_df.dropna(subset=["_pos_S", "_pos_T"])

    if not include_diagonal:
        long_df = long_df[long_df["_pos_S"] != long_df["_pos_T"]]

    if keep == "upper":
        long_df = long_df[long_df["_pos_S"] < long_df["_pos_T"]]
    elif keep == "lower":
        long_df = long_df[long_df["_pos_S"] > long_df["_pos_T"]]
    elif keep == "all":
        pass
    else:
        raise ValueError("keep must be one of: 'upper', 'lower', 'all'")

    long_df = long_df.drop(columns=["_pos_S", "_pos_T"]).sort_values(
        ["theme_id_S", "theme_id_T"], ignore_index=True
    )

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(output_csv_path, index=False, float_format=float_format)

    return long_df



# ================================================== #
# ===================== PLOTTING =================== #
# ================================================== #


# GRADIANTS

# colors=["#ffffff",
#         "#4B5D6E",
#         "#2c3e50",
#         "#2980b9",
#         "#6fa1d3", 
#         "#27ae60",
#         "#ffd200",
#         "#e74c3c"]

# colors = ["#ffffff",
#         "#4B5D6E",
#         "#2c3e50",
#         "#ffd200",
#         "#6fa1d3",
#         "#0f4a85",
#         "#27ae60",
#         "#e74c3c"]


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
            hovertemplate="Image_Theme: %{y}<br>Caption_Theme: %{x}<br>" + value_col + ": %{z}<extra></extra>",
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
            hovertemplate="Image_Theme: %{y}<br>mean " + value_col + ": %{z}<extra></extra>",
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
            hovertemplate="Caption_Theme: %{x}<br>mean " + value_col + ": %{z}<extra></extra>",
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


def theme_id_heatmap(
    df: pd.DataFrame,
    theme_col: str = "theme_id",
    title: str = "theme_id cosine similarity matrix with averages",
    colorscale: str = "Viridis",
    reverse_colors: bool = False,
    show_values: bool = False,
    width: int = 900,
    avg_gap: float = 0.06,  # vertical spacing between main heatmap and column averages
):
    """
    Build a heatmap plus row/col averages. Figure width is set by `width`;
    height matches to keep the main matrix square. `avg_gap` adds breathing
    room between the main grid and the bottom averages.
    """
    data = df.copy()
    if theme_col in data.columns:
        data = data.set_index(theme_col)

    z = data.to_numpy()
    row_labels = data.index.astype(str)
    col_labels = data.columns.astype(str)

    row_avg = z.mean(axis=1, keepdims=True)
    col_avg = z.mean(axis=0, keepdims=True)

    text_main = [[f"{v:.2f}" for v in row] for row in z] if show_values else None
    text_row = [[f"{v[0]:.2f}"] for v in row_avg] if show_values else None
    text_col = [[f"{v:.2f}" for v in col_avg[0]]] if show_values else None

    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.85, 0.15],
        row_heights=[0.85, 0.15],
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "heatmap"}, None]],
        horizontal_spacing=0.02,
        vertical_spacing=avg_gap,
    )

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=col_labels,
            y=row_labels,
            text=text_main,
            texttemplate="%{text}" if show_values else None,
            hovertemplate="Row %{y}, Col %{x}: %{z:.3f}<extra></extra>",
            coloraxis="coloraxis",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=row_avg,
            x=["row avg"],
            y=row_labels,
            text=text_row,
            texttemplate="%{text}" if show_values else None,
            hovertemplate="Row %{y} avg: %{z:.3f}<extra></extra>",
            coloraxis="coloraxis",
        ),
        row=1, col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=col_avg,
            x=col_labels,
            y=["col avg"],
            text=text_col,
            texttemplate="%{text}" if show_values else None,
            hovertemplate="Col %{x} avg: %{z:.3f}<extra></extra>",
            coloraxis="coloraxis",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=title,
        coloraxis=dict(
            colorscale=colorscale,
            reversescale=reverse_colors,
            colorbar=dict(title="Value"),
        ),
        width=width,
        height=width,  # keep main heatmap square-ish
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(matches="y1", showticklabels=False, row=1, col=2)
    fig.update_xaxes(matches="x1", showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=1)

    return fig