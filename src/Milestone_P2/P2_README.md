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

4. Context analysis:<br>
   4.a) How do external factors influence Humor ? In particular, can we link trends found in the dataset to societal landmarks ?

<!-- Mettre une funiness analysis for images => justifier pourquoi certaine images ou + ou - de votes que d'autres. -->

---

## Proposed Additional Datasets
At first, we though about enriching our dataset with some event-based data in order to conduct  the contextual analysis exposed above. In particular, we thought about using the GDELT dataset. 
The GDELT project is a large open database of human society, for the past 215 years. One can filter the data by location, type of events (meaning protests, policy change, conflict…) to make it more relevant in our case. It also includes a Goldstein scale, which is a measure of not only impact but also positivity/negativity, ranging from -10 to +10.

In order to answer out last research question, we have started to namely look into 

**The GDELT Events dataset** is available at [this link](https://www.gdeltproject.org).
We believed this would allow us to explore how real-world events and global sentiment during those years may have influenced the themes and tone of humor in the contest’s captions. We wanted to use this dataset for our context analysis. It is provided in an csv format and can also be accessed through the Google BigQuery API for efficient querying. It contains structured event-level data with attributes such as actors, event type, location, and media tone. 
There are 17 features in the dataset, here are some relevant ones

| **Features** | **Description** |
|-----------|--------------|
| Date | The date at which th event took place |
| Target | The actor (country or region) which is the recipient of that event|
| QuadClass| Simpllified categories of events based on their type of interaction|
| Goldstein | A numeric score assigned to each event to measure its impact on political stability |

However, after diving deeply into the website and available data, we not only found out that the data size was unmanageable (one month = 2GB of data), but also that we did not have access to Google Quiery, which was the most optimized way to access the data. It would also require an extensive amount of data wrangling to extract relevant information from the GDELT dataset which would not be realistic considering the other tasks we aim to tackle.
This could have become a whole project, which could have been really interesting to only based on this data augmentation. It would have required us to only focus on this questions and disregard everything else which was mentioned above. For this reason, we chose to solely focus on the other research questions.

Although we do not have additional datasets, we are still enriching our data with some augmentation methods explained below.


## Methods
<!-- We started with preprocessing. merged files.  -->
 All methods started with the same root : pre-procesing and data-cleaning. <!--We tried to merge all csv's, but came to the conclusion that it was not useful. -->
We also handled all the missing data. Namely, contest 525 and 540 are not included into the dataset. It also lacks image descriptions from contest id 764 until the end - id 895 which we suplemented by LLM augmentation via Ollama.

After these pre-processing steps, here are the respective methods used to tackle our research questions (please refer to section [Research Questions](#research-questions)).

1. Theme Analysis<br>
   1.a) What themes have a tendency to be ranked better ? <br>

   To extract themes from captions and image decriptions, we will use the _Bidirectional Encoder Representations from Transformers_ or _BERT_ which is an extensively trained NLP by Google. More specifically, we are going to take advantage of the _BERTopic_

   1.b) How theme relationships work between image and their text prompt (one way, or two-way) ?

2. Structural Analysis <br>
   2.a) Keep it short and simple: Are shorter jokes actually funnier ?<br>
   2.b) Are jokes with specific punctuation and rythms more effective ?
   
3. Semantic Analysis <br>
   3.a) What type of jokes over or underperform ? More specifically, which of the following categories perform best : dark humor, irony, social commentary, wordplay or absurdity ?
   
4. Context Analysis<br>
   4.a) How do external factors influence Humor ? In particular, can we link trends found in the dataset to societal landmarks ? 

<!-- Mettre une funiness analysis for images => justifier pourquoi certaine images ou + ou - de votes que d'autres. -->

This question was meant to be answered using an additional dataset, the GDELT one. However, upon navigating not only their Website interface but also their dataset, we have realized that, although it does give insight into relevant societal landmarks (such as armed conflicts, protests etc), it is way too big to be properly processed. 

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
