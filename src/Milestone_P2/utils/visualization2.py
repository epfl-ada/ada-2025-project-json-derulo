import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from math import floor
import matplotlib.pyplot as plt
import plotly.express as px 
from pathlib import Path
from matplotlib.image import imread


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
        filename = f"{contest_num}.csv"
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
    
    