# General
import os

# data manipulation
import json
import pandas as pd

# Visualization
from visualization import plot_df_histogram, plot_df_two_class, plot_df_pairgrid

# Themes extraction
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

# Load the JSON file
file_path = "../../data/newyorker_caption_contest/contests.json"
with open(file_path, "r") as file:
    contests = json.load(file)
    
# Convert to DataFrame
contests_df = pd.DataFrame(contests)

# Convert metadata column to columns: num_captions, num_votes, image_locations, 
# image_descriptions, image_uncanny_descriptions, entities, questions
# Note: the last 5 columns are lists (may be empty, 1 or more elements)
metadata_df = contests_df["metadata"].apply(pd.Series)
contests_df = pd.concat([contests_df.drop("metadata", axis=1), metadata_df], axis=1)

# Force dtype conversions (note: num_captions and num_votes contain NaNs => Int64 type allow NA)
contests_df["contest_id"] = contests_df["contest_id"].astype(int)
contests_df["num_captions"] = contests_df["num_captions"].astype("Int64")
contests_df["num_votes"] = contests_df["num_votes"].astype("Int64")

# contests_df.head()

# _ = plot_df_two_class(contests_df, "num_captions", "num_votes", trace="none", figsize=(6,5))

# rdm sample texts
texts = ["This is a political caption.",
         "This is a funny caption.",
         "This is a caption about animals.",
         "This is a caption about technology.",
         "This is a caption about sports.",
         "This is a caption about food.",
         "This is a caption about travel.",
         "This is a caption about health.",
         "This is a caption about education.",
         "This is a caption about environment."]

emb_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Debug 1")
# BERTopic (tiny-corpus friendly)
topic_model = BERTopic(
    embedding_model=emb_model,          # <-- provide embedding model
    representation_model=KeyBERTInspired(),
    min_topic_size=2,                   # small dataset
    nr_topics=None,                     # avoid brittle topic reduction on tiny data
    verbose=True
)
print("Debug 2")
# Fit (let BERTopic compute embeddings internally)
topics, probs = topic_model.fit_transform(texts)
print("Debug 3")
# Labels
labels = topic_model.get_topic_info()[["Topic", "Name", "Count"]]
print(labels)


# OUTPUT:
# ================================================================================================ #
# Debug 1
# Debug 2
# 2025-10-25 02:10:57,242 - BERTopic - Embedding - Transforming documents to embeddings.
# Batches: 100%|████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.23it/s]
# 2025-10-25 02:10:58,076 - BERTopic - Embedding - Completed ✓
# 2025-10-25 02:10:58,076 - BERTopic - Dimensionality - Fitting the dimensionality reduction algorithm
# OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
# 2025-10-25 02:11:00,470 - BERTopic - Dimensionality - Completed ✓
# 2025-10-25 02:11:00,470 - BERTopic - Cluster - Start clustering the reduced embeddings
# 2025-10-25 02:11:00,471 - BERTopic - Cluster - Completed ✓
# 2025-10-25 02:11:00,472 - BERTopic - Representation - Fine-tuning topics using representation models.
# 2025-10-25 02:11:01,007 - BERTopic - Representation - Completed ✓
# Debug 3
#    Topic                                    Name  Count
# 0     -1  -1_caption_animals_education_political      5
# 1      0            0_caption_health_food_sports      3
# 2      1            1_caption_technology_this_is      2
# ================================================================================================ #