# Project Title: Quantifying Humour through Semantic and Structural Analysis

## Abstract

_"A day without laughter is a day wasted"_ - Charlie Chaplin

Each week, the NewYorker magazine provides an original cartoon, with a missing caption. Readers are thus invited to not only caption that week's contest, but also rate the previous week's image.

In this project, we have decided to deeply dive into the NewYorker's dataset, which give us great insight into humour, one of the most complex forms of human communication. In this project, we will try to scientifically quantify one of the most subjective and culturally rooted, but also universally shared, expression of human emotion : laughter. By diving deeply into this data's analysis, we aim to discover patterns, structures, and other subtleties that make something "funny". 

---

## Research Questions
Based on th objectives presented in the abstract, we aim to tackle the following concrete questions : 

1. Theme analysis:

 a) What themes have a tendency to be ranked better ?
 
 b) How theme relationships work between image and their text prompt (one way, or two-way) ?

2. Structural analysis: 

 a) Keep it short and simple: Are shorter jokes actually funnier ?
 
 b) Are jokes with specific punctuation and rythms more effective ?

3. Semantic analysis:

 a) What type of jokes over or underperform ? More specifically, which of the following categories perform best : dark humor, irony, social commentary, wordplay or absurdity ?


4. Context analysis:
   
a) How do external factors influence Humor ? In particular, can we link trends found in the dataset to societal landmarks ? 


---

## Proposed Additional Datasets
At first, we though about enriching our dataset with some event-based data in order to conduct  the contextual analysis exposed above. In particular, we thought about using the GDELT dataset. 
The GDELT project is a large open database of human society, for the past 215 years. One can filter the data by location, type of events (meaning protests, policy change, conflict…) to make it more relevant in our case. It also includes a Goldstein scale, which is a measure of not only impact but also positivity/negativity, ranging from -10 to +10.

In order to answer out last research question, we have started to namely look into 

 [The GDELT Events dataset] is available at https://www.gdeltproject.org.
We believed this would allow us to explore how real-world events and global sentiment during those years may have influenced the themes and tone of humor in the contest’s captions. We will use this dataset for our context analysis. It is provided in an csv format and can also be accessed through the Google BigQuery API for efficient querying. It contains structured event-level data with attributes such as actors, event type, location, and media tone, and its size is manageable via filtered API requests for specific years or regions.
There are 17 features in the dataset, here are some relevant ones

| **Features** | **Description** |
|-----------|--------------|
| Date | The date at which th event took place |
| Target | The actor (country or region) which is the recipient of that event|
| QuadClass| Simpllified categories of events based on their type of interaction|
| Goldstein | A numeric score assigned to each event to measure its impact on political stability |

However, after diving deeply into the website and available data, we not only found out that the data size was unmanageable (one month = 2GB of data), but also that we did not have access to Google Quiery, which was the most optimized way to access the data. It would also require an extensive amount of data wrangling to extract relevant information from the GDELT dataset. 
This could have become a whole project, which could have been really interesting to only based on this data augmentation. It would have required us to only focus on this questions and disregard everything else which was mentioned above. For this reason, we chose to solely focus on the other research questions.


## Methods
We started with preprocessing. merged files. 
All methods started with the same root : pre-procesing and data-cleaning. We tried to merge all csv's,but came to the conclusion that it was not useful. 
We also handled all the missing data. Namely, contest 525 and 540 are missing. We are also missing some image descriptions (from 764 until the end) which we chose to generate via ChatGPT after discussion with our TA. 

1. Theme Analysis

  a) What themes have a tendency to be ranked better ? 
   theme extractor, 
   wikipedia links
 b) How theme relationships work between image and their text prompt (one way, or two-way) ?


2. Structural Analysis
   
   a) Keep it short and simple: Are shorter jokes actually funnier ?
   
   b) Are jokes with specific punctuation and rythms more effective ?
   
4. Semantic Analysis
   
   a) What type of jokes over or underperform ? More specifically, which of the following categories perform best : dark humor, irony, social commentary, wordplay or absurdity ?
   
6. Context Analysis
   
a) How do external factors influence Humor ? In particular, can we link trends found in the dataset to societal landmarks ? 

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
- Until P2 (November 5th) :
- November 10th - November 16th :
- November 17th  - November 23rd :
- November 24th - November 30th :
- December 1st - December 7th :
- December 8th - December 14th : 
- December 15th - December 17th :

  
- **Week 1 (Project Setup):** Confirm datasets, set up GitHub repo, assign roles.  
- **Week 2 (Data Acquisition):** Fetch the csv and json files and supplementary datasets; begin cleaning and preprocessing.  
- **Week 3 (EDA & Preliminary Visualizations):** Explore patterns, identify initial trends, refine research questions.  
- **Week 4 (Model Development):** Exploring the overall dataset and dig deeper into individual captions to understand better the patterns.
- **Week 5 (Integration & Visualization):** Combine findings into interactive dashboards/maps.  
- **Week 6 (Polish & Storytelling):** Refine analysis, finalize narrative, prepare presentation.  

---

## Organization within the Team 

All team members : Data Acquisition and Cleaning (P1) :  We proposed individual problematics for the P1. We then analyzed all the ideas and picked the best ones and found a storytelling that merges them all. 

- **EDA & Research Questions Refinement (Milestone P2):** Cross-team effort  
(cest mariem, on garde cette partie ? )
- **Modeling & Analysis (Milestone P3):** Team Member C (lead), supported by Member A.  
- **Visualization & Communication:** Cross-team effort, coordinated by Member D.  
- **Documentation & Deliverables:** Shared responsibility, with Member D ensuring consistency and quality.  

---

## Questions for TAs (Optional)
1. Can we have a custom conda environment ?
3. Are the proposed research questions relevant ?
4. use wikipedia links, entitied to JSON --> pas eu el temps mais pourrait être utile pour le theme analysis
