import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

def collaborative_filtering_recommendations(data, target_user_id, top_n = 10):

  user_item_matrix = data.pivot_table(index= 'ID', 
                                      columns='ProdID', 
                                      values='Rating', 
                                      aggfunc= 'mean').fillna(0)
  
  user_similarity = cosine_similarity(user_item_matrix)
  target_user_index = user_item_matrix.index.get_loc(target_user_id)
  user_similarities = user_similarity[target_user_index]
  similar_users_indices = user_similarities.argsort()[::-1][1:]
  recommended_items = []

  for user_index in similar_users_indices:
    rated_by_similar_user = user_item_matrix.iloc[user_index]
    not_rated_by_target_user = (rated_by_similar_user!=0) & (user_item_matrix.iloc[target_user_index]==0)

    recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])
  recommended_items_details = data[data['ProdID'].isin(recommended_items)][['Name','ReviewCount','Brand','ImageURL','Rating']]
  return recommended_items_details
#Example usage
if __name__ == "__main__":
    import pandas as pd
    from preprocess_data import process_data

    raw_data = pd.read_csv("clean_data.csv")
    data = process_data(raw_data)

    user_4_data = data[data["ID"] == 17]
    print(user_4_data)

    target_user_id = 17
    print(collaborative_filtering_recommendations(data, target_user_id))