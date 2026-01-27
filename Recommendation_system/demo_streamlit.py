import streamlit as st  # 🛑 First Streamlit command EVER in this file

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Smart Recommendation System",
    layout="wide"
)

import pandas as pd
import os

from preprocess_data import process_data
from rating_based_recommendation import get_top_rated_items
from content_based_filtering import content_based_recommendation
from collaborative_based_filtering import collaborative_filtering_recommendations

# ------------------- APP TITLE -------------------
st.title("🛍️ Smart Product Recommendation System")

# ------------------- CSS STYLING -------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Poppins', 'Segoe UI', sans-serif;
}
/* Add your own styles here */
</style>
""", unsafe_allow_html=True)

# ------------------- DATA LOADING -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "clean_data.csv")

@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_PATH)
    data = process_data(raw)

    data.columns = [c.strip() for c in data.columns]
    data["Rating"] = pd.to_numeric(data["Rating"], errors="coerce").fillna(0).round(2)
    data["ImageURL"] = data["ImageURL"].astype(str)

    return data

data = load_data()

# ------------------- IMAGE HELPER -------------------
def get_first_image(url):
    placeholder = "https://via.placeholder.com/150"

    if pd.isna(url) or url.strip() == "" or url.lower() == "nan":
        return placeholder

    for sep in ["|", ",", " "]:
        if sep in url:
            for part in url.split(sep):
                part = part.strip()
                if part.startswith("http"):
                    return part

    if url.startswith("http"):
        return url

    return placeholder

# ------------------- DISPLAY CARD -------------------
def display_products(df, cols=5):
    if df is None or df.empty:
        st.warning("No products found.")
        return

    rows = (len(df) + cols - 1) // cols
    idx = 0

    for _ in range(rows):
        columns = st.columns(cols)
        for col in columns:
            if idx >= len(df):
                break

            p = df.iloc[idx]
            name = p.get("Name", "N/A")
            brand = p.get("Brand", "N/A")
            rating = p.get("Rating", 0)
            reviews = p.get("ReviewCount", 0)
            img = get_first_image(p.get("ImageURL", ""))

            with col:
                st.markdown(f"""
                <div style="text-align:center;">
                    <img src="{img}" width="140" height="140"><br>
                    <b>{name}</b><br>
                    ⭐ {rating} • {reviews} reviews<br>
                    {brand}
                </div>
                """, unsafe_allow_html=True)
            idx += 1

# ------------------- SIDEBAR -------------------
st.sidebar.header("🧑‍💻 User Options")
user_id = st.sidebar.number_input("User ID (0 = New User)", min_value=0, step=1)
product_name = st.sidebar.text_input("🔍 Search products (comma separated)")
recommend_btn = st.sidebar.button("✨ Get Recommendations")

# ------------------- HOME PAGE -------------------
if not recommend_btn:
    if user_id == 0:
        st.subheader("⭐ Top Rated Products (Trending Now)")
        display_products(get_top_rated_items(data, 10))
    else:
        st.subheader("🎯 Personalized Recommendations")
        collab_recs = collaborative_filtering_recommendations(data, user_id, 10)

        if collab_recs.empty:
            st.info("Not enough data — showing trending products instead.")
            display_products(get_top_rated_items(data, 10))
        else:
            display_products(collab_recs)

# ------------------- SEARCH / HYBRID -------------------
if recommend_btn:
    if product_name.strip():
        st.subheader("🔍 Content-Based Recommendations")
        matches = data[data["Name"].str.contains(product_name, case=False, na=False)]
        display_products(matches)

        if user_id > 0:
            st.markdown("---")
            st.subheader("✨ Collaborative Recommendations")
            collab_recs = collaborative_filtering_recommendations(data, user_id, 5)
            display_products(collab_recs)

    elif user_id == 0:
        st.subheader("⭐ Top Rated Products")
        display_products(get_top_rated_items(data, 10))

    else:
        st.subheader("🎯 Hybrid Recommendations")
        last_item = data[data["ID"] == user_id]["Name"].iloc[-1]
        content_rec = content_based_recommendation(data, last_item, 5)
        collab_rec = collaborative_filtering_recommendations(data, user_id, 5)

        final = pd.concat([content_rec, collab_rec]).drop_duplicates().head(10)
        display_products(final)

# ------------------- FOOTER -------------------
st.markdown("---")
st.caption("✨ Smart Recommendation System | Rating • Content • Collaborative • Hybrid")
