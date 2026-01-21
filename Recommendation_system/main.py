import pandas as pd
import numpy as np

from preprocess_data import process_data
from rating_based_recommendation import get_top_rated_items
from content_based_filtering import content_based_recommendation
from collaborative_based_filtering import collaborative_filtering_recommendations
from hybrid_approach import hybrid_recommendation_filtering

# Load raw data
raw_data = pd.read_csv("clean_data.csv")

# Process data
data = process_data(raw_data)

# Get top 10 rated products
rating_based_recommendation = get_top_rated_items(data, top_n=10)
print("Rating-based recommendations:")
print(rating_based_recommendation)

# To get content based recommended items
item_name = 'OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath'
content_based_recom = content_based_recommendation(data, item_name, top_n = 8)
print("\nContent-based recommendations:")
print(content_based_recom)

# To get Collaborative based recommended items
target_user_id = 4
collaborative_filtering_rec = collaborative_filtering_recommendations(data, target_user_id, top_n = 5)
print("\nCollaborative filtering recommendations:")
print(collaborative_filtering_rec)

# Hybrid recommendation
hybrid_rec = hybrid_recommendation_filtering(
    data,
    item_name=item_name,
    target_user_id=target_user_id,
    top_n=5
)

# Fallback if hybrid is empty
if hybrid_rec.empty:
    print("\nHybrid was empty, showing top-rated items instead")
    hybrid_rec = get_top_rated_items(data, top_n=5)

print("\nHybrid recommendations:")
print(hybrid_rec)