# Project Title: Quantifying Humour through Semantic and Structural Analysis

## Abstract & Project proposal

"A day without laughter is a day wasted"- Charlie Chaplin
Humour is one of the most complex forms of human communication and it can be easily be very subjective and  deeply influenced by culture and context. In this project, we will try to find out if humour can be measured and quatified using the New Yorker caption contest dataset. The contest consists of an uncaptioned cartoon published every week solliciting online readers to caption them. The dataset consists of the cartoons, captions, and ratings for each contest from 510 to 895. In our project, we will try to find the secret recipe for a funny joke, analyzing the semantics and structures of the funniest captions and dig deeper into the themes that make a joke "funny".


---

## Research Questions
1. Theme analysis:

 a) What themes have a tendency to be ranked better?
 b)How theme relationships work between image and their text prompt (one way, or two-way)

2. Structural analysis: 

 a)Keep it short and simple: Are shorter jokes funnier
 b)Are jokes with specific punctuation, rythms, etc.. more effective?

3. Semantic analysis:

 a)What type of jokes over or underperform?
 b)Can we catagorize the jokes according to their humour type ?(dark humor, irony, social commentary, wordplay, absurdity … )

4. Context analysis:
How do external factors influence Humor?

---

## Proposed Additional Datasets

 The GDELT Events dataset corresponding to the years covered by the New Yorker Caption Contest. This will allow us to explore how real-world events and global sentiment during those years may have influenced the themes and tone of humor in the contest’s captions. We will use this dataset for our context analysis. It is provided in an csv format and can also be accessed through the Google BigQuery API for efficient querying. It contains structured event-level data with attributes such as actors, event type, location, and media tone, and its size is manageable via filtered API requests for specific years or regions

## Methods

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
 -  Mathematical method based on the influence of TTR (type token ratio, #of unique words divided by the total # words) on caption's ranking to asses whether richer vocabulary jokes rank better.
---

## Proposed Timeline Example by ChatGPT
- **Week 1 (Project Setup):** Confirm datasets, set up GitHub repo, assign roles.  
- **Week 2 (Data Acquisition):** Fetch the csv and json files and supplementary datasets; begin cleaning and preprocessing.  
- **Week 3 (EDA & Preliminary Visualizations):** Explore patterns, identify initial trends, refine research questions.  
- **Week 4 (Model Development):** Exploring the overall dataset and dig deeper into individual captions to understand better the patterns.
- **Week 5 (Integration & Visualization):** Combine findings into interactive dashboards/maps.  
- **Week 6 (Polish & Storytelling):** Refine analysis, finalize narrative, prepare presentation.  

---

## Organization within the Team
- **Data Acquisition & Cleaning (Milestone P1):** All team members proposed individual problematics for the P1. We then analyzed all the ideas and picked the best ones and found a storytelling that merges them all. 
- **EDA & Research Questions Refinement (Milestone P2):** Cross-team effort  
(cest mariem, on garde cette partie ? )
- **Modeling & Analysis (Milestone P3):** Team Member C (lead), supported by Member A.  
- **Visualization & Communication:** Cross-team effort, coordinated by Member D.  
- **Documentation & Deliverables:** Shared responsibility, with Member D ensuring consistency and quality.  

---

## Questions for TAs (Optional)
1. Can we complete the missing image descriptions by ourselves ?
2. Can we have a custom conda environment ?
3. Are the proposed research questions relevant ? 
