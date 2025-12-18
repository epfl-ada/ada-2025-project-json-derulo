from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def plot_df_pairgrid_plotly_simple(df,cols,title= 'Pair Grid Plot',**kwargs):
    '''
    
    '''
    numeric_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError('No numeric columns found.')

    fig = px.scatter_matrix(df,dimensions=numeric_cols,title=title,**kwargs)

    fig.update_traces(marker=dict(size=4,color="#27AE60"))

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=740,
        height=600
    )

    return fig

def plot_histograms(
    df: pd.DataFrame,
    cols: list,
    nbins: int = 30,
    title: str = "Histograms"
):
    numeric_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    n = len(numeric_cols)

    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=numeric_cols
    )

    for i, col in enumerate(numeric_cols, start=1):
        fig.add_trace(
            go.Histogram(
                x=df[col],
                nbinsx=nbins,
                name=col,
                showlegend=False
            ),
            row=1,
            col=i
        )

    fig.update_layout(
        title=title,
        bargap=0.05,
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=740,
        height=600
    )

    return fig

def plot_dominant_vs_mean(
    df,
    categories=("Logic", "Identity", "Action", "Language"),
    title="Dominant category strength vs mean funniness"
):
    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=categories,
        shared_yaxes=True
    )

    positions = {
        "Logic": (1, 1),
        "Identity": (1, 2),
        "Action": (1, 3),
        "Language": (1, 4),
    }

    for cat in categories:
        r, c = positions[cat]

        sub = df[df["dominant_category"] == cat]

        fig.add_trace(
            go.Scatter(
                x=sub["dominant_value"],
                y=sub["mean"],
                mode="markers",
                marker=dict(size=5, opacity=0.5),
                name=cat,
                showlegend=False
            ),
            row=r,
            col=c
        )

        fig.update_xaxes(title_text="Dominant value", row=r, col=c)

    fig.update_yaxes(title_text="Mean funniness", row=1, col=1)

    fig.update_layout(
        title=title,
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=740,
        height=600
    )

    return fig

def descriptive_analysis(df):
    df_copy=df.copy()
    # Compute statistics by dominant category
    stats = df_copy.groupby('dominant_category')['mean'].agg(['mean', 'std','sem','count']).reset_index() 

    # Add additional statistics
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])  # Standard error
    stats['ci_lower'] = stats['mean'] - 1.96 * stats['sem']  # 95% CI lower
    stats['ci_upper'] = stats['mean'] + 1.96 * stats['sem']  # 95% CI upper

    print('Statistics by dominant category:')
    print(stats)
    
    fig = px.bar(stats,
        x='dominant_category',
        y='mean',
        error_y='sem',
        text='count',
        title="Mean funniness by dominant category (± SEM)",
    )
    
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis_title="Mean funniness",
        xaxis_title="Dominant category",
        width=740,
        height=600
        
    )
    fig.show()
    
    return stats    

def heatmap(df,cols):
    corr_matrix = df[cols].corr()
    print("Correlation Matrix:")
    print(corr_matrix)
    # Heatmap
    fig = px.imshow(corr_matrix,text_auto='.2f', zmin=-1, zmax=1,labels=dict(color="Correlation"),title='Correlation Heatmap')
    fig.update_layout(width=740, height=600)
    fig.show()
    
