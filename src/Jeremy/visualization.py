import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Fait avec ChatGPT 5.0 et affiné par Jeremy Serillon.
# ---------------------------------------------------------------------------

def plot_df_histogram(
    df, class_col='Species_Common_Name', top_n=None, figsize=(10, 5),
    title=None, cmap='viridis', log_scale=False, save_path=None,
    # --- tuning knobs ---
    label_pad_linear=0.01,     # frac of max count to pad above bars for labels (linear y)
    label_pad_log=0.08,        # frac of bar height to pad above bars for labels (log y)
    ylim_headroom_linear=0.06, # extra space above highest label (linear y)
    ylim_headroom_log=0.8,     # extra space above highest label (log y)
    tight_layout_pad=1.1       # padding for tight_layout
):
    """
        Description:
        Plot a histogram of class distribution in a dataframe column, with count labels above each bar.
        
        Parameters:
            - df (pandas.DataFrame):  pandas DataFrame containing the data
            - class_col (str):  column name in df containing class labels
            - top_n (int):  if specified, only plot the top N most frequent classes
            - figsize (tuple):  figure size tuple (width, height)
            - title (str):  plot title; if None, a default title is generated
            - cmap (str):  matplotlib colormap name for bar colors
            - log_scale (bool):  if True, use logarithmic scale for y-axis
            - save_path (str):  if specified, save the plot to this file path

        Additional parameters for label placement:
            - label_pad_linear (float): padding above bars for labels (linear scale)
            - label_pad_log (float): padding above bars for labels (log scale)
            - ylim_headroom_linear (float): extra space above highest label (linear scale)
            - ylim_headroom_log (float): extra space above highest label (log scale)
            - tight_layout_pad (float): padding for plt.tight_layout()
            
        This function was made with ChatGPT 5.0 and fine-tuned by Jeremy Serillon.
    """
    if class_col not in df.columns:
        raise KeyError(f"Column '{class_col}' not found in dataframe")

    # Replace empty / NaN labels
    s = df[class_col].fillna('').astype(str).replace('', 'No animal')

    class_counts = s.value_counts()
    if top_n is not None:
        class_counts = class_counts.head(top_n)

    n = len(class_counts)
    try:
        cmap_obj = plt.get_cmap(cmap)
    except Exception:
        cmap_obj = plt.get_cmap('tab20')
    colors = [cmap_obj(x) for x in np.linspace(0.0, 1.0, n)]

    fig, ax = plt.subplots(figsize=figsize)
    class_counts.plot(kind='bar', ax=ax, color=colors)
    ax.set_xlabel(class_col)
    ax.set_ylabel('Count [log]' if log_scale else 'Count')
    if log_scale:
        ax.set_yscale('log')

    max_count = float(class_counts.max())

    # Initial y-limit (rough headroom so labels don't start outside)
    if log_scale:
        ax.set_ylim(bottom=max(0.9, ax.get_ylim()[0]), top=max_count * (1.0 + max(label_pad_log, 0.15)))
    else:
        ax.set_ylim(top=max_count * (1.0 + max(label_pad_linear, 0.08)))

    # ---- annotate bars -----------------------------------------------------
    label_offset = (1.0 + label_pad_log) if log_scale else (label_pad_linear * max_count)
    texts = []
    for bar in [p for p in ax.patches if isinstance(p, Rectangle)]:
        h = bar.get_height()
        if h <= 0:
            continue
        x = bar.get_x() + bar.get_width() / 2.0
        y = h * label_offset if log_scale else h + label_offset
        # thousands separator improves readability on large counts
        t = ax.text(
            x, y, f"{int(h):,}".replace(",", "'"),  # use apostrophe as thousands separator
            ha='center', va='bottom', rotation=90, fontsize=8, clip_on=False
        )
        texts.append(t)

    # ---- expand y-limits to fit ALL labels --------------------------------
    # Need a draw pass so text bboxes exist
    fig.canvas.draw()
    # Get a renderer in a backend-agnostic way
    _get_renderer = getattr(fig.canvas, "get_renderer", None)
    if callable(_get_renderer):
        renderer = _get_renderer()
    else:
        renderer = getattr(fig.canvas, "renderer", None)
    inv = ax.transData.inverted()

    tops = []
    if renderer is not None:
        for t in texts:
            bbox_disp = t.get_window_extent(renderer=renderer)
            # top-right of text bbox -> data coords, we only need the y
            _, y_top_data = inv.transform((bbox_disp.x1, bbox_disp.y1))
            tops.append(y_top_data)
    else:
        # Fallback: use the data y-positions of the labels if no renderer is available
        for t in texts:
            _, y_pos = t.get_position()
            tops.append(y_pos)

    if tops:
        bottom, current_top = ax.get_ylim()
        needed_top = max(tops)
        if log_scale:
            new_top = needed_top * (1.0 + ylim_headroom_log)
        else:
            new_top = needed_top + max_count * ylim_headroom_linear
        ax.set_ylim(bottom, max(current_top, new_top))

    # -----------------------------------------------------------------------

    if title is None:
        title = f"{'Top ' + str(top_n) + ' ' if top_n is not None else ''}{class_col} Distribution"
    ax.set_title(title)

    plt.tight_layout(pad=tight_layout_pad)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')  # save before show

    plt.show()
    
# Example usage:
# plot_df_histogram(missing_df,
#                   class_col='Species_Common_Name',
#                   top_n=None,
#                   figsize=(6,5),
#                   title='Missing Images Distribution',
#                   cmap='viridis',
#                   log_scale=False)


# ---------------------------------------------------------------------------

def plot_df_histogram_compare(
    df_train, df_balanced_train, class_col='Species_Common_Name', top_n=None,
    figsize=(12, 5), title=None, cmap='viridis', log_scale=False, save_path=None,
    label_pad_linear=0.01, label_pad_log=0.08,
    ylim_headroom_linear=0.06, ylim_headroom_log=0.8,
    tight_layout_pad=1.1
):
    import numpy as np, matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerBase
    import matplotlib.image as mimage

    # --- gradient legend handler (for Δ swatch) ---
    class _HandlerColormap(HandlerBase):
        def __init__(self, cmap_obj, n=256, **kw):
            self.cmap = cmap_obj; self.n = n
            super().__init__(**kw)
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            grad = np.linspace(0, 1, self.n).reshape(1, -1)
            im = mimage.AxesImage(legend.axes, cmap=self.cmap, transform=trans, interpolation='nearest')
            im.set_data(grad)
            im.set_extent([xdescent, xdescent + width, ydescent, ydescent + height])
            return [im]

    # --- counts ---
    if class_col not in df_train.columns or class_col not in df_balanced_train.columns:
        raise KeyError(f"Column '{class_col}' not found in both dataframes")
    s_train = df_train[class_col].fillna('').astype(str).replace('', 'No animal')
    s_bal   = df_balanced_train[class_col].fillna('').astype(str).replace('', 'No animal')

    counts_train = s_train.value_counts()
    classes = list(counts_train.head(top_n).index) if top_n is not None else list(counts_train.index)
    counts_train = counts_train.reindex(classes, fill_value=0)
    counts_bal   = s_bal.value_counts().reindex(classes, fill_value=0)

    n = len(classes)
    if n == 0: raise ValueError("No classes to plot.")

    try: cmap_obj = plt.get_cmap(cmap)
    except Exception: cmap_obj = plt.get_cmap('tab20')
    colors = [cmap_obj(x) for x in np.linspace(0.0, 1.0, n)]

    orig  = counts_train.values.astype(float)
    bal   = counts_bal.values.astype(float)
    delta = bal - orig
    bottoms = np.where(delta > 0, orig, bal)
    heights = np.abs(delta)

    x = np.arange(n); width = 0.7
    fig, ax = plt.subplots(figsize=figsize)

    # bars
    ax.bar(x, orig,    width=width, color=(0.65, 0.65, 0.65))  # gray base (no legend entry)
    ax.bar(x, heights, width=width, bottom=bottoms, color=colors, alpha=0.95)

    # lock x so legend won't affect scaling
    ax.set_xlim(-0.5, n - 0.5)
    ax.margins(x=0.02)

    ax.set_xlabel(class_col)
    ax.set_ylabel('Count [log]' if log_scale else 'Count')
    if log_scale: ax.set_yscale('log')

    max_val = float(np.max([orig.max() if orig.size else 0, bal.max() if bal.size else 0, 1.0]))
    if log_scale:
        ax.set_ylim(bottom=max(0.9, ax.get_ylim()[0]), top=max_val * (1.0 + max(label_pad_log, 0.15)))
    else:
        ax.set_ylim(top=max_val * (1.0 + max(label_pad_linear, 0.08)))

    # annotate only delta inside the colored segment
    for i, (btm, h, d) in enumerate(zip(bottoms, heights, delta)):
        if h <= 0: continue
        ax.text(x[i], btm + h/2.0, f"{'+' if d>0 else '−'}{int(abs(d))}",
                ha='center', va='center', fontsize=9, fontweight='bold', color='black')

    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=0)

    # legend: ONLY the gradient entry
    gradient_proxy = Patch(facecolor='none', label='Added/Removed images (Δ)')
    ax.legend(handles=[gradient_proxy],
              handler_map={gradient_proxy: _HandlerColormap(cmap_obj)},
              loc='upper right', frameon=False)

    if title is None: title = "Class Distribution (Balanced vs Original)"
    ax.set_title(title)

    plt.tight_layout(pad=tight_layout_pad)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
# plot_df_histogram_compare(train_data, new_train_data,
#                           class_col='Species_Common_Name',
#                           top_n=None, figsize=(5,5),
#                           title="Added or removed images to create balanced training set",
#                           cmap='viridis', log_scale=False)


# ---------------------------------------------------------------------------

def visualize_species_split(
    train_data, val_data, test_data, class_col='Species_Common_Name',
    top_n=None, figsize=(10, 5), title=None, cmap='viridis', log_scale=False, save_path=None,
    # tuning knobs to match your style
    label_pad_linear=0.01, label_pad_log=0.08,
    ylim_headroom_linear=0.06, ylim_headroom_log=0.8,
    tight_layout_pad=1.1
):
    """
    Plot one stacked bar per split (Training/Validation/Testing),
    stacked by species distribution (same colormap/label vibe as plot_df_histogram).
    """
    # --- clean & count -------------------------------------------------------
    def _counts(df):
        if class_col not in df.columns:
            return pd.Series(dtype=int)
        return (df[class_col].fillna('').astype(str).replace('', 'No animal')
                .value_counts())

    c_train = _counts(train_data)
    c_val   = _counts(val_data)
    c_test  = _counts(test_data)

    # union of all classes
    all_classes = pd.Index(sorted(set(c_train.index) | set(c_val.index) | set(c_test.index)))
    # totals across splits
    totals = c_train.reindex(all_classes, fill_value=0) + \
             c_val.reindex(all_classes,   fill_value=0) + \
             c_test.reindex(all_classes,  fill_value=0)

    if len(all_classes) == 0:
        raise ValueError("No classes found in any split.")

    # pick order (top_n by total, rest collapsed as 'Other' to keep legend tidy)
    if top_n is not None and top_n < len(all_classes):
        top_classes = list(totals.sort_values(ascending=False).head(top_n).index)
        other_name = "Other"
        # re-map counts into top classes + 'Other'
        def _collapse(s):
            top = s.reindex(top_classes, fill_value=0)
            other = int(s.sum() - top.sum())
            if other > 0:
                top.loc[other_name] = other
            return top
        c_train_p = _collapse(c_train)
        c_val_p   = _collapse(c_val)
        c_test_p  = _collapse(c_test)
        plot_classes = list(c_train_p.index)  # top + maybe 'Other' at end
    else:
        # no collapse, keep all (ordered by total desc for nicer legend)
        plot_classes = list(totals.sort_values(ascending=False).index)
        c_train_p = c_train.reindex(plot_classes, fill_value=0)
        c_val_p   = c_val.reindex(plot_classes,   fill_value=0)
        c_test_p  = c_test.reindex(plot_classes,  fill_value=0)

    n = len(plot_classes)

    # --- colors (same approach) ---------------------------------------------
    try:
        cmap_obj = plt.get_cmap(cmap)
    except Exception:
        cmap_obj = plt.get_cmap('tab20')
    colors = [cmap_obj(x) for x in np.linspace(0.0, 1.0, max(n, 2))][:n]

    # --- stack data in split order ------------------------------------------
    sets = ["Training set", "Validation set", "Testing set"]
    data = [
        c_train_p.values.astype(float),
        c_val_p.values.astype(float),
        c_test_p.values.astype(float),
    ]
    totals_per_set = [d.sum() for d in data]

    # --- plotting ------------------------------------------------------------
    x = np.arange(3)
    width = 0.7

    fig, ax = plt.subplots(figsize=figsize)

    bottoms = np.zeros(3, dtype=float)
    bars_by_class = []

    for cls_idx, cls in enumerate(plot_classes):
        heights = np.array([data[0][cls_idx], data[1][cls_idx], data[2][cls_idx]], dtype=float)
        bars = ax.bar(x, heights, width=width, bottom=bottoms, color=colors[cls_idx], label=cls)
        bars_by_class.append(bars)
        bottoms += heights

    ax.set_xticks(x, sets)
    ax.set_xlabel("Split")
    ax.set_ylabel('Count [log]' if log_scale else 'Count')
    if log_scale:
        ax.set_yscale('log')

    max_count = float(max(totals_per_set + [1.0]))

    # initial headroom like your function
    if log_scale:
        ax.set_ylim(bottom=max(0.9, ax.get_ylim()[0]), top=max_count * (1.0 + max(label_pad_log, 0.15)))
    else:
        ax.set_ylim(top=max_count * (1.0 + max(label_pad_linear, 0.08)))

    # --- annotate totals at bar tops ----------------------------------------
    label_offset = (1.0 + label_pad_log) if log_scale else (label_pad_linear * max_count)
    texts = []
    for xi, tot in zip(x, totals_per_set):
        if tot <= 0:
            continue
        y = tot * label_offset if log_scale else tot + label_offset
        t = ax.text(xi, y, f"{int(tot):,}".replace(",", "'"),
                    ha='center', va='bottom', rotation=90, fontsize=8, clip_on=False)
        texts.append(t)

    # --- expand y-limits to fit labels (renderer-aware) ---------------------
    fig.canvas.draw()
    _get_renderer = getattr(fig.canvas, "get_renderer", None)
    renderer = _get_renderer() if callable(_get_renderer) else getattr(fig.canvas, "renderer", None)
    inv = ax.transData.inverted()

    tops = []
    if renderer is not None:
        for t in texts:
            bbox_disp = t.get_window_extent(renderer=renderer)
            _, y_top_data = inv.transform((bbox_disp.x1, bbox_disp.y1))
            tops.append(y_top_data)
    else:
        for t in texts:
            _, y_pos = t.get_position()
            tops.append(y_pos)

    if tops:
        bottom, current_top = ax.get_ylim()
        needed_top = max(tops + [ax.get_ylim()[1]])
        new_top = (needed_top * (1.0 + ylim_headroom_log)) if log_scale else (needed_top + max_count * ylim_headroom_linear)
        ax.set_ylim(bottom, max(current_top, new_top))

    # --- legend --------------------------------------------------------------
    ax.legend(loc='upper right', frameon=False, ncols=1)

    if title is None:
        title = "Species distribution across splits"
    ax.set_title(title)

    plt.tight_layout(pad=tight_layout_pad)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
# Example usage:
# visualize_species_split(new_train_data, val_data, test_data,
#                         class_col='Species_Common_Name',
#                         top_n=None, figsize=(5, 5),
#                         title="Train/Val/Test Species Split",
#                         cmap='viridis', log_scale=False)


# ---------------------------------------------------------------------------