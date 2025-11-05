# Project Title: Quantifying Humour through Semantic and Structural Analysis

## Abstract

_"A day without laughter is a day wasted"_ - Charlie Chaplin

Each week, the NewYorker magazine provides an original cartoon, with a missing caption. Readers are thus invited to not only caption that week's contest, but also rate the previous week's image. 

In recent years, meme culture, and internet culture overall, has consistently prioritized shorter, and trending jokes. Tweets, TikToks, and even indirectly related formats, are tending to become shorter and have visual support. This is one of the many factors that could define how an image supported joke lands.

In this project, we have decided to dive into the NewYorker's dataset, which will give us great insights into humour, one of the most complex forms of human communication. We aim to discover the themes, structures, semantics, context and other subtleties that make something "funny".

---

## Research Questions
Based on th objectives presented in the abstract, we aim to tackle the following concrete questions : 

1. Theme analysis:<br>
   1.a) What themes have a tendency to be ranked better ?<br>
   1.b) How theme relationships work between image and their text prompt (one way, or two-way) ?

2. Structural analysis: <br>
   2.a) Keep it short and simple: Are shorter jokes actually funnier ?<br>
   2.b) Are jokes with specific punctuation and rythms more effective ?

3. Semantic analysis:<br>
   3.a) What type of jokes over or underperform ? More specifically, which of the following categories perform best : dark humor, irony, social commentary, wordplay or absurdity ?

4. Context analysis: [this reasearch question was dropped, see below why]<br>
   4.a) How do external factors influence Humor ? In particular, can we link trends found in the dataset to societal landmarks ?

<!-- Mettre une funiness analysis for images => justifier pourquoi certaine images ou + ou - de votes que d'autres. -->

---

## Proposed Additional Datasets
At first, we though about enriching our dataset with some event-based data in order to conduct the contextual analysis exposed above. We thought about using the [GDELT dataset](https://www.gdeltproject.org). 

The GDELT project is a large open database of human society, for the past 215 years. There are 17 features in the dataset, here are the most relevant ones to us:

| **Features** | **Description** |
|-----------|--------------|
| _Date_ | The date at which th event took place |
| _Target_ | The actor (country or region) which is the recipient of that event |
| _QuadClass_| Simplified categories of events based on their type of interaction (e.g. 'protests', 'policy change', 'conflict', …)|
| _Goldstein_ | Numeric score assigned to each event to measure its impact on political stability |

We believed this would allow us to explore how real-world events and global sentiment during those years may have influenced the themes and tone of humor in the contest’s captions.

However, after diving deeply into the available data, we found out that the data size was unmanageable (one month of data = 2GB while our dataset spans from 2016 to 2024). and it would also require an extensive amount of data wrangling which would not be realistic considering the other tasks we aim at tackling. For this reason, we chose to solely focus on the other research questions.

Although we do not have additional datasets, we are still enriching our data with some augmentation methods explained below.


## Methods
 All methods started with the same root : pre-procesing and data-cleaning (missing data, corrupted captions, LLM augmentation via Ollama, ...). 

After pre-processing, here are the respective methods used to tackle our [research qestions](#research-questions).

1. Theme Analysis<br>
   For 1.a) and 1.b), here are the main tools and techniques used in the theme extraction pipeline:
   - Pre-process to keep only meaningful words ("This is politic." => "politic")
   - Sentence Embeddings (BERT / Sentence-BERT / CLIP text encoder) - converts text => semantic vector representation.
   - Spherical k-means - clusters embeddings to form our themes.
   - c-TF-IDF (**c**lass-based **T**erm **F**requency-**I**nverse **D**ocument **F**requency) - extracts and labels the most representative words for each cluster/theme.
   - (previously tested) BERTopic - topics clustering approach that inspired the pipeline but is less scalable and less controllable for large datasets such as ours.

   We will extract our themes from our decscriptions/captions like so:
   - images descriptions processed all at once to allow clustering them to each other independantly from the captions.
   - captions processed for each contest internally: extract the themes of all the caption for all specific contest once at a time.
   
   This allows us to analyse and discuss behaviours inside each contest and also to parts the computation. To do more general analysis across ALL the captions, we will then do additional clustering to group back all the intermidiate, contest-wise, results.

   To assess whether image and caption themes are statistically associated, we will construct a contingency table of their co-occurrences and apply chi-square test or pointwise mutual information (PMI/NPMI) to identify theme pairs that occur significantly more or less often.


2. Structural Analysis <br>
   To investigate whether joke structure affects performance, we can analyse two textual aspects: length and punctuation/rhythm.
   For each caption, we extract structural features such as the number of words (joke length) and punctuation counts (e.g., exclamation marks, question marks, commas, and sentence breaks as indicators of rhythm). This will allow us to estimate the rhythm of a sentence as well as the average sentence length. We also use the _Type Token Ratio_, _TTR_ for short, which is the number of unique words / total words used to measure the richness of vocabulary

   We will then examine how these features relate to the mean performance metric using correlation analyses (e.g. through Spearman’s correlation coefficient). This allows us to test whether shorter jokes or those with distinct punctuation patterns tend to achieve higher performance.
   

3. Semantic Analysis <br>
   The aim is to quantitatively compare the performance of jokes across humor categories.
   Performance is measured using the mean metric provided in the dataset for each caption, which accounts for both the number of votes and the funniness-score of the proposal.

   The main steps are as follows:
   - Categorize the jokes into five humor types—dark humor, irony, social commentary, wordplay, and absurdity—based on the framework from Deceptive Humor: A Synthetic Multilingual Benchmark Dataset for Bridging Fabricated Claims with Humorous Content.
   - Compute the average performance metric for each category.
   - Statistically test differences in performance between categories using methods such as one-way ANOVA. This will help us determine how statistically relevant and significant our results are.
   
4. Context Analysis<br>
   Question 4.a) was meant to be answered using an additional dataset, the GDELT dataset. However, upon navigating not only their Website interface but also their dataset, we have realized that, although it does give insight into relevant societal landmarks (such as armed conflicts, protests etc), it is way too big to be properly processed, as explained earlier. 

- **Data Collection & Cleaning:** 
 -Preprocessing and more calculations in order to associate a date to the contest.
 -Merging all the csv's in on big csv with caption numbers.
 -Handeling the missing data like missing contests (525 and 540), missing image descriptions (from 764), missimg contest dates
  
- **Exploratory Data Analysis (EDA):** 
 - Visualization through all contests
 - A more precise visualization for each contest

- **Modeling:**  (c Mariem: on mets quoi ici ?)
  - LLM for semantic analysis
  - Time series models (e.g., ARIMA, Prophet) for predicting demand and congestion.  
  - Machine learning (e.g., random forests, gradient boosting) for multi-factor prediction.  

- **Semantic and Structural Analysis** 
 - data handling, cleaning up of the semantics
 - DHD
 -  Mathematical method based on the influence of TTR (type token ratio, #of unique words divided by the total # words) on caption's ranking to asses whether richer vocabulary jokes rank better.
---

## Proposed Timeline 
- **Until P2 (November 5th) :** This time was spent settig up of our Github page, understanding, but also pre-processing and cleaning our data. We have also performed preliminary analysis as well as looked into enriching our data. 
- **November 10th - November 16th :** We will be refining our research question. What exactly is feasible ? 
- **November 17th  - November 23rd :** We will be exploring and diving deeper into our refinedresearch questions, identifying key patterns and trends.
- **November 24th - November 30th :** We will be finalizing our analysis, or at least have rough figures.
-** December 1st - December 7th :** We will be finalizing our analysis.
-** December 8th - December 14th :** We will start to create our website. The goal is to combine our findings into interactive maps which work with a coherent and engaging datastory. 
- **December 15th - December 17th :** This corresponds to the P3 deadline. By this time, we should only have relatively small changes to do. Mostly, cleaning up the latest bugs and correcting what needs to be fixed after meeting with our TA.  

We also have a specific time-slot which we have alloted every week in order to give each other feedback as well as keep up with what the others are doing. Our weekly meeting takes place every wednesdays from 10 until 12.

---

## Organization within the Team 

- All team members : Data Acquisition and Cleaning (P1) :  We proposed individual problematics for the P1. We then analyzed all the ideas and picked the best ones and found a storytelling that merges them all. 

- Marina and Jeremy : These members are in charge of semantics and theme analysis. They not only found out tools but also different libraries which can be used in order to further analyze this part.

- Hanna looked into the GDELT dataset. This includes browsing the website, looking into ehat each index means, as well as visualizing the data. This was also supported by Mariem and Jeremy.
  
- Adrian

- Mariem


---

## Questions for TAs (Optional)
1. We would want to also incorporate the wikipedia links, which can be in the contests.json file, to our semantics analysis. The goal is to find recurring themes. Would that be feasible ? 
4. We will slowly start looking into website design. We have mostly heard of Jerkyll.
