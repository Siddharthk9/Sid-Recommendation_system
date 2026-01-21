import pandas as pd
import numpy as np

from preprocess_data import process_data
from rating_based_recommendation import get_top_rated_items
from content_based_filtering import content_based_recommendation
from collaborative_based_filtering import collaborative_filtering_recommendations

def hybrid_recommendation_filtering(
    data: pd.DataFrame,
    item_name: str,
    target_user_id: int,
    top_n: int = 10):

    content_based_rec = content_based_recommendation(data, item_name, top_n)
    collaborative_based_rec = collaborative_filtering_recommendations(data, target_user_id, top_n)
  
 # Combine & remove duplicates
    hybrid_approach = (
        pd.concat([content_based_rec, collaborative_based_rec])
        .drop_duplicates()
        .head(2*top_n)
    )
    return hybrid_approach

#To test
if __name__ == "__main__":
    import pandas as pd
    raw_data = pd.read_csv("clean_data.csv")
    data = process_data(raw_data)
    
    from preprocess_data import process_data

    #item_name = "OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath"
    item_name = "Pure Gold Bitter Orange Essential Oil, 100% Natural & Undiluted, 60ml"
    target_user_id = 5
    top_n = 5

    hybrid_rec = hybrid_recommendation_filtering(data, item_name, target_user_id, top_n)

    print(hybrid_rec)