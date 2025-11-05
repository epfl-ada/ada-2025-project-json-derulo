# Project Title: Quantifying Humour through Semantic and Structural Analysis

## Abstract

_"A day without laughter is a day wasted"_ - Charlie Chaplin

Every week, the NewYorker magazine provides an original cartoon with a missing caption, inviting readers to assign a caption to it and to rate the previous week's caption propositions. 

In recent years, the Internet culture has consistently prioritized shorter and trending jokes: Tweets and TikToks are becoming shorter, using primarily visual support, redefining the influence of images on jokes.

In this project, we are diving into the NewYorker's dataset, which will give us insight into humour, one of the most complex forms of human communication. We aim to discover the themes, structures, semantics, context and other subtleties that make something "funny".

---

## Research Questions
We aim to tackle the following questions: 

1. Theme analysis:<br>
   1.a) What themes have a tendency to be ranked better?<br>
   1.b) How do theme relationships work between image and their text prompt (one way, two-way)?

2. Structural analysis: <br>
   2.a) Are shorter jokes funnier?<br>
   2.b) Are jokes with specific punctuation, vocabulary and rythm more effective?

3. Semantic analysis:<br>
   3.a) What type of jokes over or underperform ? Which of these categories perform best: dark humor, irony, social commentary, wordplay or absurdity?

4. Context analysis: [dropped reasearch question]<br>
   4.a) How do external factors influence humor? Can we link trends in the dataset to societal landmarks?

<!-- Mettre une funiness analysis for images => justifier pourquoi certaine images ou + ou - de votes que d'autres. -->

---

## Proposed Additional Datasets
At first, we thought about enriching our dataset with event-based data to conduct the contextual analysis exposed above by using the [GDELT dataset](https://www.gdeltproject.org). 

The GDELT project is a large open database of human society, from the past 215 years, and it is composed of 17 features. Here are the most relevant ones to us:

| **Features** | **Description** |
|-----------|--------------|
| _Date_ | When the event took place |
| _Target_ | The actor which is the recipient of that event |
| _QuadClass_| Simplified categories of events based on their type of interaction (e.g. 'protests', 'policy change', …)|
| _Goldstein_ | Numeric score assigned to each event for its impact on political stability |

We believed this would allow us to explore how real-world events and public sentiment during those years may have influenced the themes and tone of humor in the contests.

However, we found that the data size was unmanageable (one month of data = 2GB), and it would also require an extensive amount of data wrangling which would not be realistic considering the other tasks we aim to tackle. We chose to solely focus on the other research questions.

Although we do not have additional datasets, we are enriching our data through augmentation methods explained below.


## Methods
 All methods started with the same root: pre-procesing and data-cleaning (missing data, corrupted captions, LLM augmentation via Ollama, ...). 

Here are the methods used for our [research qestions](#research-questions):

1. Theme Analysis<br>
   For 1.a)&b), here are the main tools and techniques used for theme extraction:
   - Pre-process to keep only meaningful words ("This is politic." => "politic")
   - Sentence Embeddings (BERT / Sentence-BERT / CLIP text encoder) - converts text => semantic vector representation.
   - Spherical k-means - clusters embeddings to form our themes.
   - c-TF-IDF (**c**lass-based **T**erm **F**requency-**I**nverse **D**ocument **F**requency) - extracts and labels the most representative words for each cluster/theme.
   - BERTopic - topics clustering approach, but is less scalable and less controllable for large datasets.

   The themes are extracted as such:
   - images descriptions are processed all at once, allowing them to cluster with each other independently from the captions.
   - captions are processed for each contest internally and the themes extracted once at a time.
   
   This allows us to analyse and discuss behaviours inside each contest and to part the computation. For a general analysis across ALL the captions, we will do additional clustering to group back all the intermidiate contest-wise results.

   To assess whether image and caption themes are statistically associated, we will do a contingency table of their co-occurrences and apply chi-square test or pointwise mutual information (PMI/NPMI) to identify theme pairs that occur more or less often.


2. Structural Analysis <br>
   To investigate whether joke structure affects performance, we can analyse length and punctuation/rhythm. We also use the _Type Token Ratio_, _TTR_ to measure the vocabulary richness, which is: (#unique_words) / (total_words).

   We will then examine how these features relate to the mean performance metric using correlation analyses, testing whether shorter jokes or those with punctuation patterns achieve higher scores.
   

3. Semantic Analysis <br>
   Aim: quantitatively compare the performance of jokes across humor categories.
   Performance is measured using the mean metric provided, which accounts for the number of votes and the funniness-score of the proposal.

   The main steps are:
   - Categorize the jokes into five humor types (dark humor, irony, social commentary, wordplay, absurdity) based on the framework from deceptive humor
   - Compute average performance score for each category.
   - Statistically test differences in performance between categories using methods like one-way ANOVA.


- **Data Collection & Cleaning:** 
 - Preprocessing and calculations to associate a date to the contest.
 - Merging all the csv's in one with caption numbers.
 - Handling missing data like missing contests (525 and 540), missing image descriptions (from 764), missimg contest dates
  
- **Exploratory Data Analysis (EDA):** 
 - Visualization through all contests
 - Precise visualization for each contest

- **Modeling:**  
  - LLM for semantic analysis
  - Time series models (ARIMA, Prophet) for predicting demand and congestion.  
  - Machine learning (random forests, gradient boosting) for multi-factor prediction.  

- **Semantic and Structural Analysis** 
 - data handling, semantics cleaning 
 - DHD
 - TTR: asses influence of vocabulary richness.
---

## Proposed Timeline 
- **Until P2 (November 5th):** settig up of our Github page, understanding, pre-processing and cleaning our data. We have also performed preliminary analysis and looked into enriching our data. 
- **November 10th - November 16th:** refining our research question: what exactly is feasible? 
- **November 17th  - November 23rd:** diving deeper into our refined research questions, identifying key patterns and trends.
- **November 24th - November 30th:** finalizing our analysis, or have rough figures.
- **December 1st - December 7th:** finalization of our analysis.
- **December 8th - December 14th:** creation of our website. 
- **December 15th - December 17th:** cleaning up the latest bugs and correcting what needs to be fixed.

We have a weekly meetings together for feedback and to keep up with what the others are doing. 

## Organization within the Team 

- All team members: data acquisition and cleaning (P1) and proposition of individual problematics: we then picked the best ideas and found a storytelling that merges them. 

- Marina and Jeremy: in charge of semantics and theme analysis, finding tools and different libraries which can be used further analysis.

- Hanna: looked into the GDELT dataset: browsing the website, looking into what each index means, and data visualisation. This was supported by Mariem and Jeremy.
  
- Adrian: extensive sanity check, adding missing metadata to json file.

- Mariem: data enrichment, preprocessing and merging: includes doing a sanity check of the json_file metadata and adding the missing dates, num_captions and num_votes on the json metadata. 


---

## Questions for TAs 
1. We would want to also incorporate the wikipedia links, which can be in the contests.json file, to our semantics analysis. Can we feasibly find recurring themes. Would that be feasible ? 
4. We will start looking into website design.
