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
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import nbimporter
from scipy.ndimage import gaussian_filter1d
    
import os, base64, mimetypes

from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

from typing import Optional

from IPython.display import display, Markdown, HTML
from ipywidgets import IntSlider, VBox, Output



def plot_df_pairgrid_plotly(df,cols,title= 'Pair Grid Plot'):
    '''
    Takes a dataframe and makes a pairwise scatter plot with the given columns
    df: dataframe with the data
    cols: columns of intersts that are ploted (as a list e.g ["mean", "Action"])
    title : Title of the figure
    
    returns the figure
    '''
    numeric_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError('No numeric columns found.')

    fig = px.scatter_matrix(df,dimensions=numeric_cols,title=title)
    fig.update_traces(marker=dict(size=4,color='#27AE60'))
    fig.update_layout(plot_bgcolor='white',paper_bgcolor='white',width=740,height=600)
    return fig

def plot_histograms(df,cols,nbins= 30,title='Histograms'):
    '''
    Takes a dataframe and makes histograms with the given columns
    df: dataframe with the data
    cols: columns of intersts that are ploted
    title : Title of the figure
    
    returns the figure
    '''
    numeric_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    n = len(numeric_cols)

    fig = make_subplots(rows=1,cols=n,subplot_titles=numeric_cols)
    for i, col in enumerate(numeric_cols, start=1):
        fig.add_trace(go.Histogram(x=df[col],nbinsx=nbins,name=col,showlegend=False),row=1,col=i)
        
    fig.update_layout(title=title,bargap=0.05,plot_bgcolor='white',paper_bgcolor='white',width=740,height=600)
    return fig

def plot_dominant_vs_mean(df,title='Dominant category strength vs mean funniness'):
    '''
    Takes dataframes with columns containing the dominant category and dominant 
    value and makes scatter plot of the dominant categories vs the mean
    df: dataframe with the data
    title : Title of the figure
    
    returns the figure
    '''
    categories=('Logic', 'Identity', 'Action', 'Language')
    fig = make_subplots(rows=1,cols=4,subplot_titles=categories,shared_yaxes=True)

    positions = {'Logic': (1, 1),'Identity': (1, 2),'Action': (1, 3),'Language': (1, 4),}
    for cat in categories:
        r, c = positions[cat]
        sub = df[df['dominant_category'] == cat]
        
        fig.add_trace(go.Scatter(x=sub['dominant_value'],y=sub['mean'],mode='markers',marker=dict(size=5, opacity=0.5),
                name=cat,showlegend=False),row=r,col=c)
        
        fig.update_xaxes(title_text='Dominant value', row=r, col=c)
        
    fig.update_yaxes(title_text='Mean funniness', row=1, col=1)
    fig.update_layout(title=title,plot_bgcolor='white',paper_bgcolor='white',width=740,height=600)
    return fig

def descriptive_analysis(df):
    '''
    Takes a dataframe and computes, prints and plot (bar plot) basic statistics such as mean, standard deviations ...
    df : Dataframe with the columns domionant_category 
    
    returns a data frame with the basic statistics
    '''
    df_copy=df.copy()
    # Compute statistics by dominant category
    stats = df_copy.groupby('dominant_category')['mean'].agg(['mean', 'std','sem','count']).reset_index() 

    # Add additional statistics
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])  # Standard error
    stats['ci_lower'] = stats['mean'] - 1.96 * stats['sem']  # 95% CI lower
    stats['ci_upper'] = stats['mean'] + 1.96 * stats['sem']  # 95% CI upper

    print('Statistics by dominant category:')
    print(stats)
    
    fig = px.bar(stats,x='dominant_category',y='mean',error_y='sem',text='count',title='Mean funniness by dominant category ± SEM')
    
    fig.update_layout(plot_bgcolor='white',paper_bgcolor='white',yaxis_title='Mean funniness',xaxis_title='Dominant category',
                      width=740,height=600)
    fig.show()
    return stats    

def heatmap(df,cols):
    '''
    Takes a dataframe and computes, print and plot the correlation matrix with the given columns
    df: dataframe with the data
    cols: the columns that will be take into account for the computation
    
    return nothing
    '''
    corr_matrix = df[cols].corr()
    print('Correlation Matrix:')
    print(corr_matrix)
    
    # Heatmap
    fig = px.imshow(corr_matrix,text_auto='.2f', zmin=-1, zmax=1,labels=dict(color='Correlation'),title='Correlation Heatmap')
    fig.update_layout(width=740, height=600)
    fig.show()
    

def plot_divised_scatter_votes(df,figsize=(20, 15)):
    '''
    df: Pandas dataframe for 1 contest (csv), containing funny, somewhat funny, not funny
    
    figsize= figure size
    '''
    x = np.arange(len(df))
    plt.rcParams["figure.figsize"] = figsize  
    votes=['funny','somewhat_funny','not_funny']
    colors=['b','r','g']
    
    for i,vote in enumerate(votes):
        
        plt.subplot(3,1,i+1)
        plt.scatter(x,df[vote],label=f'{vote} votes',color=colors[i])
        maximum_value=max(df[vote])
        plt.ylim(top=1.1*maximum_value)
        if df[vote].iloc[0]>100:
            plt.yscale("log")
        plt.title(f'{vote} votes vs caption')
        plt.xlabel('captions')
        plt.ylabel(f'number of {vote} votes')
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=16)
        
    plt.suptitle('Precise votes distributions',fontsize=20)
    plt.tight_layout()
    plt.show()
    
def plot_divised_histo_stacked(df,division=3,n_caption_per_div=20,figsize=(12,6)):
    '''
    df: dataframe from 1 contes (csv) containing funny, somewhat funny, not funny
    
    division: the number of subplots you want
    
    n_caption_per_divion: the number of caption per subplots
    
    figsize= figure size
    '''
    plt.rcParams["figure.figsize"] = figsize  
    separation=floor(len(df)/(division-1))
    frame=[0,n_caption_per_div]
    
    for div in range(division):
        window=df.iloc[frame[0]:frame[1]]
        
        plt.subplot(1,division,div+1)
        plt.bar(window.index,window['not_funny'], color='green', label='Not funny votes')
        plt.bar(window.index,window['somewhat_funny'], bottom=window['not_funny'],color='red', label='Somewhat funny votes')
        plt.bar(window.index,window['funny'],bottom=(window['not_funny']+window['somewhat_funny']), color='blue', label='funny votes')
        
        maximum_value=max(window['not_funny']+window['somewhat_funny']+window['funny'])
        plt.ylim(top=1.1*maximum_value)
        
        plt.title(f'Captions {frame[0]}–{frame[1]} ({n_caption_per_div} shown)')
        plt.xlabel('captions')
        plt.ylabel('Number of votes')
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        
        start = frame[0] + separation - round(n_caption_per_div / 2)
        end = frame[1] + separation + round(n_caption_per_div / 2)
        start = max(0, start)
        end = min(len(df), end)
        frame = [start, end] 
    plt.suptitle('Divised bar plot')
    plt.tight_layout()
    plt.show()
    
def scatter_error(df,figsize=(12,6)):
    '''
    df: dataframe from 1 contest (csv) containing the mean and precision of each quote
    
    figsize= figure size
    '''
    plt.figure(figsize=figsize)
    plt.errorbar(df.index, df['mean'], yerr=df['precision'], fmt='o', ecolor='r', capthick=2,label='Mean with error bars')
    plt.title('Error bars for the means')
    plt.xlabel('Captions')
    plt.ylabel('Mean funny votes')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_divised_scatter_error(df,division=3,n_caption_per_div=20,figsize=(12,6)):
    '''
    df: dataframe from 1 contes (csv) containing the mean and precision of each quote
    
    division: the number of subplots you want
    
    n_caption_per_divion: the number of caption per subplots
    
    figsize= figure size
    '''
    
    plt.rcParams["figure.figsize"] = figsize  
    separation=floor(len(df)/(division-1))
    frame=[0,n_caption_per_div]
    for div in range(division):
        window=df.iloc[frame[0]:frame[1]]
        
        plt.subplot(1,division,div+1)
        plt.errorbar(window.index, window['mean'], yerr=window['precision'], fmt='o', ecolor='r', capthick=2,label='Mean with error bars')
        
        maximum_value=max(window['mean'])
        plt.ylim(top=1.1*maximum_value)
        
        plt.title(f'Captions {frame[0]}–{frame[1]} ({n_caption_per_div} shown)')
        plt.xlabel('captions')
        plt.ylabel('Means')
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        
        start = frame[0] + separation - round(n_caption_per_div / 2)
        end = frame[1] + separation + round(n_caption_per_div / 2)
        start = max(0, start)
        end = min(len(df), end)
        frame = [start, end] 
    plt.suptitle('Divised scatter',fontsize=15)
    plt.tight_layout()
    plt.show()

def box_plot(df,figsize=(12,6)):
    '''
    df: dataframe from 1 contes (csv) containing the mean of each quote
    
    figsize= figure size
    '''
    plt.figure(figsize=figsize)
    plt.boxplot(df['mean'],orientation='horizontal',vert=False, patch_artist=True)
    plt.title('Boxplot of the means')
    plt.tight_layout()
    plt.show()
    
    
def plot_funny_dist(root):
    funny_distribution=pd.DataFrame({'contest_id':[0],'funny':[0]})
    for file in os.listdir(root):
        if file.endswith(".csv"):
            contest_id = file.removesuffix(".csv")
            contest_df = pd.read_csv(os.path.join(root, file))
            funny=0
            for funny_votes in contest_df['funny']:
                if funny_votes>0:
                    funny+=1
            temporary=pd.DataFrame({'contest_id':[int(contest_id)],'funny':[funny]})
            funny_distribution = pd.concat([funny_distribution, temporary], ignore_index=True)
            

    funny_distribution=funny_distribution.drop(index=0)
    funny_distribution=funny_distribution.sort_values('contest_id')
    fig = px.scatter(
        funny_distribution,
        x='contest_id',        # or 'contest_num' if numeric
        y='funny',
        hover_data=['contest_id', 'funny'],  # info shown on hover
        title='Number of funny captions per contest',
        labels={'contest_id': 'Contest', 'funny': 'Funny captions'},
    )

    fig.update_traces(marker=dict(size=8, color='blue', opacity=0.7))
    fig.update_layout(width=1200, height=600)
    fig.show()
    return funny_distribution


def plot_funny_dist(root):
    """
    root: Path type that points where the csv files are
    """
    root = Path(root)  # ensure root is a Path object
    funny_distribution = pd.DataFrame({'contest_id': [0], 'funny': [0]})

    for file in root.glob("*.csv"):  # Path-friendly iteration
        contest_id = file.stem  # removes '.csv' automatically
        contest_df = pd.read_csv(file)

        # Count captions with at least 1 funny vote
        funny = (contest_df['funny'] > 0).sum()

        temporary = pd.DataFrame({'contest_id': [int(contest_id)], 'funny': [funny]})
        funny_distribution = pd.concat([funny_distribution, temporary], ignore_index=True)

    # Remove dummy first row
    funny_distribution = funny_distribution.drop(index=0)
    funny_distribution = funny_distribution.sort_values('contest_id')

    # Plot
    fig = px.scatter(
        funny_distribution,
        x='contest_id',
        y='funny',
        hover_data=['contest_id', 'funny'],
        title='Number of funny captions per contest',
        labels={'contest_id': 'Contest', 'funny': 'Funny captions'},
    )
    fig.update_traces(marker=dict(size=8, color='blue', opacity=0.7))
    fig.update_layout(width=1200, height=600)
    fig.show()

    return funny_distribution


def show_contest_image(contest_num,images_folder, title_suffix=""):
    image_path = os.path.join(images_folder, f"{contest_num}.jpg")
    
    if os.path.exists(image_path):
        img = imread(image_path)
        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Contest #{contest_num} {title_suffix}")
        plt.show()
    else:
        print(f"Image for contest #{contest_num} not found.")
        
def print_top_caption(contest_num, folder_path):
    filename = f"{contest_num}.csv"
    file_path = os.path.join(folder_path, filename)
    
    df = pd.read_csv(file_path)
    top1 = df.nlargest(1, 'mean')  # top caption by mean score
    
    print(f"\nContest {contest_num} Top Caption:")
    for i, row in top1.iterrows():
        print(f"Rank 1: '{row['caption']}' with mean score {row['mean']:.3f}")
        

def plot_number_of_captions_vs_contests(folder_path,image_path):
    folder_path=str(folder_path)
    contest_counts = {} #intialise caption number at 0
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}") #missing contests
    for contest_num in range(510, 896):
        filename = f"{contest_num}_cleaned.csv" #added _cleaned to make it work in new dataset
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                contest_counts[contest_num] = len(df) #to get the length of the file = number of captions per contest
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"File not found: {filename}")

    # Turning them into Dataframes
    counts_df = pd.DataFrame(list(contest_counts.items()), columns=["Contest", "Num_Captions"])
    counts_df = counts_df.sort_values(by="Num_Captions", ascending=False) #ordered list of the contest with their corresponding number of captions

    # printing the gist of the top and least populars
    print("\nTop 5 most popular contests:")
    print(counts_df.head())

    print("\nTop 5 least popular contests:")
    print(counts_df.tail())

    # Defining the top and least popular
    most_popular = counts_df.iloc[0]
    least_popular = counts_df.iloc[-1]

    print(f"\nMost popular contest: #{most_popular['Contest']} with {most_popular['Num_Captions']} captions.")
    print(f"Least popular contest: #{least_popular['Contest']} with {least_popular['Num_Captions']} captions.")

    # Visualizing
    plt.figure(figsize=(10,5))
    plt.scatter(counts_df["Contest"], counts_df["Num_Captions"], marker='o', linestyle='-')
    # Highlight most & least popular on the scatter plot
    plt.scatter(most_popular["Contest"], most_popular["Num_Captions"], color='green', s=100, label='Most popular')
    plt.scatter(least_popular["Contest"], least_popular["Num_Captions"], color='red', s=100, label='Least popular')

    plt.annotate(f"{int(most_popular['Contest'])}",
                xy=(most_popular['Contest'], most_popular['Num_Captions']),
                xytext=(most_popular['Contest'], most_popular['Num_Captions'] + 100),
                arrowprops=dict(facecolor='green', arrowstyle='->'),
                fontsize=10)

    plt.annotate(f"{int(least_popular['Contest'])}",
                xy=(least_popular['Contest'], least_popular['Num_Captions']),
                xytext=(least_popular['Contest'], least_popular['Num_Captions'] - 100),
                arrowprops=dict(facecolor='red', arrowstyle='->'),
                fontsize=10)
    plt.title("Number of Captions per Contest")
    plt.xlabel("Contest Number")
    plt.ylabel("Number of Captions")
    plt.grid(True)
    plt.show()
            
def plot_and_show20(df,title,image_path):
    plt.figure(figsize=(10,8))
    plt.barh(df["Contest"].astype(str), df["Num_Captions"], color='skyblue')
    plt.xlabel("Number of Captions")
    plt.ylabel("Contest Number")
    plt.title('Top 20 ' +title)
    plt.tight_layout()
    plt.show()
    print('top 1 ' + title)
    show_contest_image(int(df.iloc[-1]['Contest']),image_path)
    
    
def top1_means_vs_contests(folder_path,image_path):
    #now print popularity between each contest similarly to first part but using mean scores instead
    top_n = 1  # number of top captions to consider
    top_mean_scores = {}

    for contest_num in range(510, 896):
        filename = f"{contest_num}.csv"
        file_path = os.path.join(folder_path, filename)
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if len(df) >= 1:
                    top_scores = df['mean'].nlargest(top_n)
                    top_mean = top_scores.mean()
                    top_mean_scores[contest_num] = top_mean
                else:
                    top_mean_scores[contest_num] = None
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"File not found: {filename}")

    # turning the results into DataFrame and sorting them
    top_mean_df = pd.DataFrame(list(top_mean_scores.items()), columns=["Contest", f"Top{top_n}_Mean"])
    top_mean_df = top_mean_df.dropna().sort_values(by=f"Top{top_n}_Mean", ascending=False)

    # Print top and least liked contests
    print(f"\nTop 5 contests by mean of top {top_n} captions:")
    print(top_mean_df.head())

    print(f"\nBottom 5 contests by mean of top {top_n} captions:")
    print(top_mean_df.tail())

    # Get highest and lowest mean contests
    highest_mean = top_mean_df.iloc[0]
    lowest_mean = top_mean_df.iloc[-1]

    print(f"\nHighest mean contest: #{highest_mean['Contest']} with mean score {highest_mean[f'Top{top_n}_Mean']:.3f}")
    print(f"Lowest mean contest: #{lowest_mean['Contest']} with mean score {lowest_mean[f'Top{top_n}_Mean']:.3f}")

    # visualization
    plt.figure(figsize=(10,5))
    #Highlighting the highest and lowest mean contest
    plt.scatter(highest_mean["Contest"], highest_mean[f"Top{top_n}_Mean"], color='green', s=100, label='Highest mean')
    plt.scatter(lowest_mean["Contest"], lowest_mean[f"Top{top_n}_Mean"], color='red', s=100, label='Lowest mean')
    plt.scatter(top_mean_df["Contest"], top_mean_df[f"Top{top_n}_Mean"], marker='o', linestyle='-')
    plt.annotate(f"{int(highest_mean['Contest'])}",
                xy=(highest_mean['Contest'], highest_mean[f"Top{top_n}_Mean"]),
                xytext=(highest_mean['Contest'], highest_mean[f"Top{top_n}_Mean"] + 0.05),
                arrowprops=dict(facecolor='green', arrowstyle='->'),
                fontsize=10)

    plt.annotate(f"{int(lowest_mean['Contest'])}",
                xy=(lowest_mean['Contest'], lowest_mean[f"Top{top_n}_Mean"]),
                xytext=(lowest_mean['Contest'], lowest_mean[f"Top{top_n}_Mean"] - 0.05),
                arrowprops=dict(facecolor='red', arrowstyle='->'),
                fontsize=10)

    plt.title(f"Mean Score of Top {top_n} Captions per Contest")
    plt.xlabel("Contest Number")
    plt.ylabel(f"Mean of Top {top_n} Captions")
    plt.grid(True)
    plt.show()
    
    print("Contest with the best mean score on top 1 captions : ")
    show_contest_image(int(top_mean_df.iloc[0]['Contest']),image_path)
    print_top_caption(int(top_mean_df.iloc[0]['Contest']), folder_path)
    
    print("Contest with the worst mean score on top 1 captions : ")
    show_contest_image(int(top_mean_df.iloc[-1]['Contest']),image_path)
    print_top_caption(int(top_mean_df.iloc[-1]['Contest']), folder_path)

def print_top2_captions_with_delta(contest_num, folder_path):
    filename = f"{contest_num}.csv"
    file_path = os.path.join(folder_path, filename)
    
    df = pd.read_csv(file_path)
    top2 = df.nlargest(2, 'mean')  # top 2 captions by mean score
    
    delta = top2.iloc[0]['mean'] - top2.iloc[1]['mean']
    
    print(f"\nContest {contest_num} Top 2 Captions (Delta = {delta:.3f}):")
    for i, row in top2.iterrows():
        print(f"Rank {i+1}: '{row['caption']}' with mean score {row['mean']:.3f}")
        
def computing_deltas(folder_path):
    top2_delta = {} #computing the delta between the top 2 captions
    for contest_num in range(510, 896):
        filename = f"{contest_num}.csv"
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if len(df) >= 2: #removing empty contest (if they have less than 2 captions, not sure if this is necessary though)
                    top_scores = df['mean'].nlargest(2).values #making sure it's the two highest mean
                    margin = top_scores[0] - top_scores[1]
                    top2_delta[contest_num] = margin
                else:
                    top2_delta[contest_num] = None
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            top2_delta[contest_num] = None
        return top2_delta
    
def plot_deltas(folder_path,image_path):
    margin_df = pd.DataFrame(list(computing_deltas(folder_path).items()), columns=["Contest", "Top2_Delta"])
    margin_df = margin_df.dropna()  # once again making sure our DataFrame only has real data, not sure if this is necessary
    max_contest = margin_df.loc[margin_df["Top2_Delta"].idxmax()] #Contest with the highest Delta
    min_contest = margin_df.loc[margin_df["Top2_Delta"].idxmin()] #Contest with the lowest Delta
    plt.figure(figsize=(12,6))
    plt.scatter(max_contest['Contest'], max_contest['Top2_Delta'], color='green', s=100, 
                label=f'Highest margin: Contest {int(max_contest["Contest"])}')
    plt.scatter(min_contest['Contest'], min_contest['Top2_Delta'], color='red', s=100, 
                label=f'Smallest margin: Contest {int(min_contest["Contest"])}')
    # Plotting all the deltas
    plt.scatter(margin_df["Contest"], margin_df["Top2_Delta"], color='blue', label='Other contests')

    # Highlighting highest and lowest delta
    plt.scatter(max_contest['Contest'], max_contest['Top2_Delta'], color='green', s=100, label='Highest margin')
    plt.scatter(min_contest['Contest'], min_contest['Top2_Delta'], color='red', s=100, label='Smallest margin')

    #Annotating Highest and Lowest delta
    plt.annotate(f"{int(max_contest['Contest'])}", 
                xy=(max_contest['Contest'], max_contest['Top2_Delta']),
                xytext=(max_contest['Contest'], max_contest['Top2_Delta'] + 0.05),
                arrowprops=dict(facecolor='green', arrowstyle='->'),
                fontsize=10)

    plt.annotate(f"{int(min_contest['Contest'])}", 
                xy=(min_contest['Contest'], min_contest['Top2_Delta']),
                xytext=(min_contest['Contest'], min_contest['Top2_Delta'] - 0.05),
                arrowprops=dict(facecolor='red', arrowstyle='->'),
                fontsize=10)

    plt.title("Delta Between Top 1 and Top 2 Captions per Contest")
    plt.xlabel("Contest Number")
    plt.ylabel("Score Delta (Top1 - Top2)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()  
    plt.show()
    
    print("Contest with the highest delta of scores between the top 2 captions : ")
    show_contest_image(int(max_contest['Contest']),image_path)
    print_top2_captions_with_delta(int(max_contest['Contest']), folder_path)
    
    print("Contest with the lowest delta of scores between the top 2 captions : ")
    show_contest_image(int(min_contest['Contest']),folder_path)
    print_top2_captions_with_delta(int(min_contest['Contest']), folder_path)


# This file was done at LTS5 as a Semester Project by Jeremy Serillon and refined for this project
# ---------------------------------------------------------------------------

cst_cmap_purple = ListedColormap(["#500091"])   # dark purple

def plot_df_histogram(
    df, class_col='Species_Common_Name', top_n=None, figsize=(10, 5),
    title=None, cmap=cst_cmap_purple, log_scale=False, save_path=None,
    # --- tuning ---
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
    cmap_obj = cmap

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
#                   cmap=cst_cmap_purple,
#                   log_scale=False)


# ---------------------------------------------------------------------------

def plot_df_histogram_compare(
    df_train, df_balanced_train, class_col='Species_Common_Name', top_n=None,
    figsize=(12, 5), title=None, cmap=cst_cmap_purple, log_scale=False, save_path=None,
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

    cmap_obj = cmap
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
#                           cmap=cst_cmap_purple, log_scale=False)


# ---------------------------------------------------------------------------

def visualize_species_split(
    train_data, val_data, test_data, class_col='Species_Common_Name',
    top_n=None, figsize=(10, 5), title=None, cmap=cst_cmap_purple, log_scale=False, save_path=None,
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
    cmap_obj = cmap
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
#                         cmap=cst_cmap_purple, log_scale=False)



# ============================================================================= 
# VISUALIZATION CHAPTER 1 - Interactive visualization and classical plots
# =============================================================================
def plot_df_two_class(df, x_col, y_col, figsize=(12, 6),
                      title=None, cmap=cst_cmap_purple, save_path=None,
                      trace: str = "lines"):
    """
    Plot x_col vs y_col with points colored by y (low -> high) using `cmap`.

    trace options:
      - None / "none": only points
      - "lines": connect points with gradient-colored segments (by local y)
      - "poly": polynomial fit (degree 3 by default), gradient-colored by fitted y; shows mean/std of yfit
      - "gauss": Gaussian fit y = A * exp(-(x-μ)^2/(2σ^2)) in ORANGE; shows μ and σ

    Notes:
      - x_col and y_col must be numeric (no parsing).
      - 'fit' uses degree=3 by default (capped to len(x)-1).
      - 'gauss' requires at least 3 positive y values (log-fit).
    """
    if x_col not in df.columns or y_col not in df.columns:
        missing = [c for c in (x_col, y_col) if c not in df.columns]
        raise KeyError(f"Column(s) not found: {', '.join(missing)}")

    valid = {None, "none", "lines", "poly", "gauss"}
    t = (trace if trace is None else str(trace).lower())
    if t not in valid:
        raise ValueError("trace must be one of {None, 'lines', 'fit', 'gauss'}")
    trace_mode = None if t in (None, "none", "None") else t

    data = df[[x_col, y_col]].copy().dropna()
    x = pd.to_numeric(data[x_col], errors='raise').to_numpy()
    y = pd.to_numeric(data[y_col], errors='raise').to_numpy()

    # Sort by x for sensible connections/fit
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Colormap for points / gradient lines
    cmap_obj = cmap
    norm = plt.Normalize(vmin=np.min(y), vmax=np.max(y))

    fig, ax = plt.subplots(figsize=figsize)

    # Helper for number formatting in annotations
    _fmt = lambda v: f"{v:.4g}"

    # ----- traces -----
    if trace_mode == "lines" and len(x) >= 2:
        n_sub = 32
        seg_points, seg_colors = [], []
        for xi, yi, xj, yj in zip(x[:-1], y[:-1], x[1:], y[1:]):
            xf = np.linspace(xi, xj, n_sub + 1)
            yf = np.linspace(yi, yj, n_sub + 1)
            pts = np.column_stack([xf, yf]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            seg_points.append(segs)
            seg_colors.append((yf[:-1] + yf[1:]) / 2.0)
        seg_points = np.concatenate(seg_points, axis=0)
        seg_colors = np.concatenate(seg_colors, axis=0)
        lc = LineCollection(seg_points, cmap=cmap_obj, norm=norm, linewidth=1.8, zorder=2)
        lc.set_array(seg_colors)
        ax.add_collection(lc)

    yfit = None
    deg = None

    if trace_mode == "poly" and len(x) >= 2:
        deg = int(min(3, max(1, len(x) - 1)))  # default 3, capped by data
        coeffs = np.polyfit(x, y, deg=deg)
        xfit = np.linspace(x.min(), x.max(), 600)
        yfit = np.polyval(coeffs, xfit)
        # solid orange fit line above points
        ax.plot(xfit, yfit, color='orange', linewidth=2.0, zorder=4)
        # annotate caracteristics (coefs, ...) of the fitted curve
        ax.text(0.02, 0.98, f"coeffs: " + ", ".join([_fmt(c) for c in coeffs]),
                transform=ax.transAxes, ha='left', va='top', fontsize=9,
                color='orange', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), zorder=6)

    # ----- points (always) -----
    ax.scatter(x, y, c=y, cmap=cmap_obj, norm=norm, s=28, zorder=3)

    # ----- Gaussian fit (orange, on top) -----
    if trace_mode == "gauss" and len(x) >= 3:
        mask = y > 0
        if mask.sum() >= 3:
            xg, yg = x[mask], y[mask]
            c2, c1, c0 = np.polyfit(xg, np.log(yg), 2)
            if c2 < 0:
                sigma = float(np.sqrt(-1.0 / (2.0 * c2)))
                mu = float(c1 * (sigma ** 2))
                A = float(np.exp(c0 + (mu ** 2) / (2.0 * sigma ** 2)))
                xfit = np.linspace(x.min(), x.max(), 600)
                yfit_g = A * np.exp(-((xfit - mu) ** 2) / (2.0 * sigma ** 2))
                ax.plot(xfit, yfit_g, color='orange', linewidth=2.5, zorder=5)
                # annotate μ and σ
                ax.text(0.02, 0.98, f"μ = {_fmt(mu)}\nσ = {_fmt(sigma)}",
                        transform=ax.transAxes, ha='left', va='top', fontsize=9,
                        color='orange', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
                        zorder=6)

    # ----- labels, limits, title -----
    def _ordinal(n: int) -> str:
        return f"{n}{'st' if n%10==1 and n%100!=11 else 'nd' if n%10==2 and n%100!=12 else 'rd' if n%10==3 and n%100!=13 else 'th'}"

    base_title = title or f"{y_col} vs {x_col}"
    if trace_mode == "poly":
        base_title += f" (with pol. curve {_ordinal(deg)} deg)"
    elif trace_mode == "gauss":
        base_title += " (with Gaussian fit)"
    ax.set_title(base_title)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)

    if len(x) > 0:
        xr = (x.max() - x.min()) or 1.0
        yr = (y.max() - y.min()) or 1.0
        ax.set_xlim(x.min() - 0.02 * xr, x.max() + 0.02 * xr)
        ax.set_ylim(y.min() - 0.02 * yr, y.max() + 0.02 * yr)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return pd.DataFrame({x_col: x, y_col: y})


def plot_df_pairgrid(
    df: pd.DataFrame,
    cols: list,
    trace: str = "None",              # {None/'none','lines','fit','gauss'}
    cmap = cst_cmap_purple,
    figsize=(12, 12),
    include_upper: bool = False,       # plot both triangles if True; otherwise only lower
    diagonal: str = "name",            # {'name','blank'} what to show on diagonal cells
    save_path: str | None = None,
):
    """
    Plot a grid of subplots for all (x,y) pairs from `cols`, using a style consistent
    with plot_df_two_class: points colored by y (low->high) and optional trace.

    Parameters
    ----------
    df : DataFrame
    cols : list[str]
        Numeric columns to cross-plot.
    trace : {None, 'none', 'lines', 'fit', 'gauss'}
    cmap : str
    figsize : tuple
        Overall figure size.
    include_upper : bool
        If False, only the lower triangle (i>j). If True, both triangles.
    diagonal : {'name','blank'}
        What to draw on the diagonal cells.
    save_path : str | None
        If provided, saves the figure to this path.

    Returns
    -------
    dict
        Mapping (x_col, y_col) -> plotted row count for that cell.
    """

    # --- validate & prepare ---
    if not cols:
        raise ValueError("`cols` must be a non-empty list of column names.")
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in dataframe.")

    # normalize trace
    valid_trace = {None, "none", "lines", "fit", "gauss"}
    t = (trace if trace is None else str(trace).lower())
    if t not in valid_trace:
        raise ValueError("trace must be one of {None, 'lines', 'fit', 'gauss'}")
    trace_mode = None if t in (None, "none") else t

    # Colormap object (fallback if invalid)
    cmap_obj = cmap

    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False)
    out_counts = {}

    def _plot_xy_on_ax(ax, x_col, y_col):
        """Mirror of plot_df_two_class behavior, rendered onto a given Axes."""
        # Clean numeric data
        data = df[[x_col, y_col]].copy().dropna()
        try:
            x = pd.to_numeric(data[x_col], errors="raise").to_numpy()
            y = pd.to_numeric(data[y_col], errors="raise").to_numpy()
        except Exception:
            ax.text(0.5, 0.5, "non-numeric", ha="center", va="center", transform=ax.transAxes)
            return 0

        if len(x) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            return 0

        # sort by x for sensible connections/fit
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # color normalization per-plot (same as plot_df_two_class)
        norm = plt.Normalize(vmin=np.min(y), vmax=np.max(y))

        # ----- traces -----
        if trace_mode == "lines" and len(x) >= 2:
            n_sub = 32
            seg_points, seg_colors = [], []
            for xi, yi, xj, yj in zip(x[:-1], y[:-1], x[1:], y[1:]):
                xf = np.linspace(xi, xj, n_sub + 1)
                yf = np.linspace(yi, yj, n_sub + 1)
                pts = np.column_stack([xf, yf]).reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                seg_points.append(segs)
                seg_colors.append((yf[:-1] + yf[1:]) / 2.0)
            seg_points = np.concatenate(seg_points, axis=0)
            seg_colors = np.concatenate(seg_colors, axis=0)
            lc = LineCollection(seg_points, cmap=cmap_obj, norm=norm, linewidth=1.2, zorder=2)
            lc.set_array(seg_colors)
            ax.add_collection(lc)

        if trace_mode == "fit" and len(x) >= 2:
            deg = int(min(3, max(1, len(x) - 1)))
            coeffs = np.polyfit(x, y, deg=deg)
            xfit = np.linspace(x.min(), x.max(), 400)
            yfit = np.polyval(coeffs, xfit)
            pts = np.column_stack([xfit, yfit]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            y_mid = (yfit[:-1] + yfit[1:]) / 2.0
            lc = LineCollection(segs, cmap=cmap_obj, norm=norm, linewidth=1.6, zorder=4)
            lc.set_array(y_mid)
            ax.add_collection(lc)
            # annotate mean/std of fitted curve (compact)
            ax.text(0.02, 0.98,
                    f"μ={np.mean(yfit):.3g}, σ={np.std(yfit):.3g}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=7, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
                    zorder=6)

        if trace_mode == "gauss" and len(x) >= 3:
            mask = y > 0
            if mask.sum() >= 3:
                xg, yg = x[mask], y[mask]
                c2, c1, c0 = np.polyfit(xg, np.log(yg), 2)
                if c2 < 0:
                    sigma = float(np.sqrt(-1.0 / (2.0 * c2)))
                    mu = float(c1 * (sigma ** 2))
                    A = float(np.exp(c0 + (mu ** 2) / (2.0 * sigma ** 2)))
                    xfit = np.linspace(x.min(), x.max(), 400)
                    yfit_g = A * np.exp(-((xfit - mu) ** 2) / (2.0 * sigma ** 2))
                    ax.plot(xfit, yfit_g, color="orange", linewidth=1.8, zorder=5)
                    ax.text(0.02, 0.98,
                            f"μ={mu:.3g}, σ={sigma:.3g}",
                            transform=ax.transAxes, ha="left", va="top",
                            fontsize=7, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
                            zorder=6)

        # ----- points (always) -----
        ax.scatter(x, y, c=y, cmap=cmap_obj, norm=norm, s=12, zorder=3)

        # tidy axes
        xr = (x.max() - x.min()) or 1.0
        yr = (y.max() - y.min()) or 1.0
        ax.set_xlim(x.min() - 0.02 * xr, x.max() + 0.02 * xr)
        ax.set_ylim(y.min() - 0.02 * yr, y.max() + 0.02 * yr)
        return len(x)

    # --- build grid ---
    for i, y_col in enumerate(cols):
        for j, x_col in enumerate(cols):
            ax = axes[i, j]

            if i == j:
                if diagonal == "name":
                    ax.text(0.5, 0.5, y_col, ha="center", va="center", fontsize=8, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # skip upper triangle
            if not include_upper and j > i:
                ax.axis("off")
                continue
            
            # plot the (x_col, y_col) pair
            npts = _plot_xy_on_ax(ax, x_col, y_col)
            out_counts[(x_col, y_col)] = npts

            # labels only on left column and bottom row (to reduce clutter)
            if j == 0:
                ax.set_ylabel(y_col, fontsize=8)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            if i == n - 1:
                ax.set_xlabel(x_col, fontsize=8)
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return out_counts

def _img_to_data_uri(path: str) -> str:
    """Return a data URI for the image at 'path'."""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def render_contest(
    contests_df: pd.DataFrame,
    file_path: str,
    contest_id: int,
    width_px: int = 450,
    out: Optional[Output] = None,
) -> None:
    """
    Render contest details for a given contest_id into Jupyter output.

    Parameters
    ----------
    contests_df : pd.DataFrame
        DataFrame containing contests with at least columns:
        ['contest_id', 'image', 'data', 'metadata'] (metadata may be dict or None).
    file_path : str
        Path to the JSON file used to resolve image relative paths.
    contest_id : int
        The contest id to render.
    width_px : int, default 450
        Fixed width in pixels for the displayed image.
    out : ipywidgets.Output, optional
        If provided, output is written inside this Output widget.
    """
    def _do_render():
        contest = contests_df[contests_df["contest_id"] == contest_id]
        if contest.empty:
            display(Markdown("**Contest not found.**"))
            return

        contest_data = contest.iloc[0]
        display(Markdown(f"### Contest ID: {contest_data['contest_id']}"))
        display(Markdown(f"**Image Path:** {contest_data['image']}"))
        display(Markdown(f"**Data Path:** {contest_data['data']}"))

        metadata = contest_data.get("metadata")
        if isinstance(metadata, dict) and metadata:
            display(Markdown(f"**Number of Captions:** {metadata.get('num_captions', 'N/A')}"))
            display(Markdown(f"**Number of Votes:** {metadata.get('num_votes', 'N/A')}"))
            display(Markdown(f"**Image Locations:** {', '.join(metadata.get('image_locations', []) or [])}"))
            display(Markdown(f"**Image Descriptions:** {', '.join(metadata.get('image_descriptions', []) or [])}"))
            display(Markdown(f"**Entities:** {', '.join(metadata.get('entities', []) or [])}"))
        else:
            display(Markdown("**Metadata:** None"))

        # Display the image with a fixed width using HTML <img>
        image_path = os.path.join(os.path.dirname(file_path), contest_data["image"])
        if os.path.exists(image_path):
            src = _img_to_data_uri(image_path)
            display(HTML(
                f"<img src='{src}' "
                f"style='width:{int(width_px)}px;height:auto;display:block;image-rendering:auto;'/>"
            ))
        else:
            display(Markdown("**Image not found at the specified path.**"))

    if out is None:
        _do_render()
    else:
        out.clear_output(wait=True)
        with out:
            _do_render()

def make_contest_viewer(
    contests_df: pd.DataFrame,
    file_path: str,
    width_px: int = 450,
) -> VBox:
    """
    Create an interactive viewer (slider + output) for browsing contests.

    Returns
    -------
    VBox
        A widget you can display in Jupyter with `display(widget)`.
        The widget exposes `widget.slider` for reading/updating the selected id,
        and `widget.output` for the Output area.
    """
    if "contest_id" not in contests_df.columns:
        raise ValueError("contests_df must contain a 'contest_id' column.")

    # Ensure integer ids
    min_id = int(contests_df["contest_id"].min())
    max_id = int(contests_df["contest_id"].max())

    slider = IntSlider(
        min=min_id,
        max=max_id,
        step=1,
        value=min_id,
        description="Contest ID",
        continuous_update=False,
    )

    out = Output()

    def _on_change(change):
        if change["name"] == "value":
            render_contest(
                contests_df=contests_df,
                file_path=file_path,
                contest_id=int(change["new"]),
                width_px=width_px,
                out=out,
            )

    slider.observe(_on_change, names="value")

    # Initial render
    render_contest(
        contests_df=contests_df,
        file_path=file_path,
        contest_id=int(slider.value),
        width_px=width_px,
        out=out,
    )

    widget = VBox([slider, out])
    widget.slider = slider
    widget.output = out
    widget.width_px = width_px
    return widget
