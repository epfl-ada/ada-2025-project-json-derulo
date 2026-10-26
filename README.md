
# Project Applied Data Analysis - Group 'JSON DERULO'

This repository holds the project of group **JSON DERULO** for the Applied Data Analysis EPFL cours.

Group members
- Adrián Fernández Lojo - adrian.fernandezlojo@epfl.ch
- Hanna Zwygart - hanna.zwygart@epfl.ch
- Jeremy Alexandre Serillon - jeremy.serillon@epfl.ch
- Mariem Maazoun - mariem.maazoun@epfl.ch
- Marina-Carmen Meredith Romero - marina.meredith@epfl.ch

 # How to be funny, for dummies ?

## Quantifying Humour through Theme, Structural, Semantic and Temporal Analysis  

## Abstract

_"A day without laughter is a day wasted"_ - Charlie Chaplin

Every week, the NewYorker magazine provides a cartoon with a missing caption, inviting readers to assign a caption and to rate the previous week's captions. 

In recent years, the Internet culture has prioritized shorter and trending jokes: Tweets and TikToks are becoming shorter, using primarily visual support, redefining the influence of images on jokes.

In this project, we are diving into the NewYorker's dataset, which will give us insight into humour, one of the most complex forms of human communication. We aim to discover the themes, structures, semantics, and other subtleties that make jokes "funny".

## Data story 
A nice presentation of the project is available on the JSON-DERULO website: [https://hzwygart.github.io/ada-2025-project-json-derulo/](https://hzwygart.github.io/ada-2025-project-json-derulo/)



## Research Questions
We aim to tackle the following questions: 

1. Theme analysis:<br>
   1.a) How do theme relationships work between image and their text prompt ? <br>
   1.b) Are some images themes inherently funnier ? <br>
   1.c) Are semantically distant image–caption theme pairs (lower cosine similarity / higher distance) 
      - i funnier and/or 
      - ii less/more frequent than semantically similar pairs?

2. Structural analysis: <br>
   2.a) Are shorter jokes funnier?<br>
   2.b) Are jokes with specific vocabulary more effective?

3. Semantic analysis:<br>
   3.a) Which of Berger Humor's categories perform best ?

4. Temporal analysis : <br>
   4.a) What patterns emerge when visualizing timing-relation questions accross time ?

## Enrich Datasets
At first, we thought about enriching our dataset with event-based data to conduct the contextual analysis exposed above by using the [GDELT dataset](https://www.gdeltproject.org). However, we found that the data size was unmanageable (one month of data = 2GB), and it would also require an extensive amount of data wrangling which would not be realistic. We chose to solely focus on the other research questions.

Although we do not have additional datasets, we enrich the data with the contest publication dates based on this [blog](https://attemptedbloggery.blogspot.com/]), and enrich the dataset with other algorithms based on the already present data.

## Methods
 All methods started with the same root: pre-procesing and data-cleaning (missing data, corrupted captions, LLM augmentation via Ollama, ...). 

Here are the methods used for our [research qestions](#research-questions):

- **Data Collection & Cleaning:** 
 Preprocessing and more calculations in order to associate a date to the contest. 
   - Handeling the missing data like missing contests (525 and 540), missing image descriptions (from 764), missing contest dates.
   - Handeling the bad formating of csv and the missing values. The rows of csv with bad formating were removed as there is sufficient and representative data without them.
  
- **Exploratory Data Analysis (EDA):** 
   - Visualization through all contests.
   - Precise visualization for each contest.

- **Data Augmentation:**  
  - LLM for missing metada image related (i.e Image description, entities, ...) 
  - Basic algorithms for dates

- **Theme Analysis** <br>
   *Main tools and techniques used for theme extraction:*
   - Pre-process to keep only meaningful words ("This is politic." => "politic")
   - Sentence Embeddings (BERT / Sentence-BERT / CLIP text encoder) - converts text => semantic vector representation.
   - Unsupervised clustering (MiniBatchKMeans) to form our themes.
   - c-TF-IDF (**c**lass-based **T**erm **F**requency-**I**nverse **D**ocument **F**requency) - extracts and labels the most representative words for each cluster/theme.
   - LLMs for label generation

   *Analysis methods:*
   - 1.a) Spearman correlation; Pearson correlation
   - 1.b) OLS regression (fixed effects); Mixed-effects model; Wald/LR test; FDR correction
   - 1.c) OLS regression; Negative Binomial regression; Interaction regression


- **Structural Analysis** <br>
   - String manipulation
   - To investigate whether joke structure affects performance, we can analyse length and punctuation/rhythm. We also use the _Type Token Ratio_, _TTR_ to measure the vocabulary richness, which is: $\frac{\#\text{unique\_words}}{\text{total\_words}}$.
   -  We will then examine how these features relate to the mean performance metric using correlation analyses, testing whether shorter jokes or those with specific words achieve higher scores.
   - Basic statistics and a bit more advance such as correlation and pearson scores
   

- **Semantic Analysis** <br>
   - Categorize the jokes into the Berger's humor categories based on the framework from deceptive humor with the help of LLMs
   - Statistically test differences in performance, based on the caption mean, between categories using basics statistics and regression.

- **Temporal Analysis** <br>
   - Basic data comparison.


## Usage
### Package installation
From the project root, please run:
```
conda env create -f environment.yml
```
### Utilisation
Once the environment has been created, activate it and then you can go through the notebook. Some of the developped functions using Ollam need extra steps (see Below), but none of them are directly used in the notebook.
### Using LLMs

#### Using Ollama Cloud
1. Obtain an API key from [Ollama API](https://ollama.com/settings/keys).

2. The following functions need the user to put the key: LIST <br>
In those functions modify the below line with the key in a string format

```python
OPENAI_API_KEY = "YOUR_API_KEY"
```

#### Downloading Ollama's model
The function that makes the images descriptions uses llava:7b. Ollama should have been installed with the *requirements*. In order to use *add_image_descriptions()* follow the next steps:
1. Open a terminal and activate the environment
2. run the following command to download the model (4,7 GB) (You can click [here](https://ollama.com/library/llava) to see the model documentation):
```
ollama pull lava:7b
```
3. once it has be download run that will activate Ollama:
```
ollama serve
```
4. You can use the function !!


## Project Structure: where are Milestone P2 and P3?

```
.
├── README.md
├── data/
│   ├── IMPORTANT_README.md
│   ├── newyorker_caption_contest/
│   └── papers
└── src
    ├── Milestone_P2/
    └── Milestone_P3/
```
The content of our milestones can be found as indicated above. The README for each milestone sits in their respective folders.

.

.

.


### Dataset Download URL
[https://drive.google.com/file/d/1Es1XmglK5isHZuLiO2RDQfc1tYrDYzXz/view?usp=sharing](https://drive.google.com/file/d/1Es1XmglK5isHZuLiO2RDQfc1tYrDYzXz/view?usp=sharing)

#### Reference
Zhang, J.*, Jain, L.*, Guo, Y.*, Chen, J., Zhou, K. L., Suresh, S., ... & Nowak, R. (2024). Humor in AI: Massive Scale Crowd-Sourced Preferences and Benchmarks for Cartoon Captioning. arXiv preprint arXiv:2406.10522.

### Relevant papers
- [Humor in AI (PDF)](data/papers/Humor%20in%20AI%20-%20Paper.pdf) --> [URL to article](https://arxiv.org/abs/2406.10522)
- [Do Androids Laugh at Electric Sheep (PDF)](data/papers/Do%20Androids%20Laugh%20at%20Electric%20Sheep%20-%20Paper.pdf) --> [URL to article](https://aclanthology.org/2023.acl-long.41/)

### Other related links
- [https://nextml.github.io/caption-contest-data/](https://nextml.github.io/caption-contest-data/)
- [https://huggingface.co/datasets/jmhessel/newyorker_caption_contest](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest)
 

## VERY IMPORTANT: DATASET PLACEMENT
Download the dataset (see URL below), unzip/extract it, and place the resulting folder inside the ```data/``` directory.
The folder name MUST remain exactly: **```newyorker_caption_contest```**

Do NOT rename this folder. If you change the name, the dataset may get added or saved in the GitHub repo (which is not ideal because the data set is a lot of storage) and code expecting data/newyorker_caption_contest will fail.


Your ```data/``` folder should look like this:
```
data/
├── IMPORTANT_README.md
├── newyorker_caption_contest
│   ├── contests.json
│   ├── data/
│   └── images/
└── papers
    ├── Do Androids Laugh at Electric Sheep - Paper.pdf
    └── Humor in AI - Paper.pdf
```
In VS Code the folder ```newyorker_caption_contest``` should be grayed out showing that git will not track it, like shown below:

![Expected data directory layout](/figures/documentation/data_folder.png)
