import pandas as pd

def get_top_rated_items(data:pd.DataFrame, top_n:int = 10) -> pd.DataFrame:
    """
    Returns top N products based on average rating.
    """
    average_ratings = data.groupby(['Name', 'ReviewCount', 'Brand', 'ImageURL'])['Rating'].mean().reset_index()

    top_rated_items = average_ratings.sort_values(by='Rating',ascending=False)

    return top_rated_items.head(top_n)


# Get top rated items
if __name__ == "__main__":
    import pandas as pd
    from preprocess_data import process_data

    raw_data = pd.read_csv("clean_data.csv")
    data = process_data(raw_data)

    print(get_top_rated_items(data))