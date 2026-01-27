import streamlit as st

# -------------------------------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# -------------------------------------------------
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

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🛍️ Smart Product Recommendation System")

# -------------------------------------------------
# CSS
# -------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Poppins', 'Segoe UI', sans-serif;
}

.product-card {
    background: linear-gradient(145deg, #ffffff, #f3f7ff);
    border-radius: 20px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 8px 22px rgba(0,0,0,0.12);
    transition: 0.3s;
}

.product-card:hover {
    transform: translateY(-8px) scale(1.04);
    box-shadow: 0 16px 32px rgba(0,0,0,0.18);
}

.product-img {
    width: 140px;
    height: 140px;
    border-radius: 50%;
    object-fit: cover;
    border: 4px solid #ff7a00;
    margin-bottom: 12px;
}

.badge-rating {
    background: #00c853;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
}

.badge-review {
    background: #2962ff;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
}

h2, h3 {
    color: #ff6a00;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# DATA
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "clean_data.csv")

@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_PATH)
    data = process_data(raw)
    data.columns = [c.strip() for c in data.columns]
    data["Rating"] = pd.to_numeric(data["Rating"], errors="coerce").fillna(0)
    data["ImageURL"] = data["ImageURL"].astype(str)
    return data

data = load_data()

# -------------------------------------------------
# IMAGE HANDLER
# -------------------------------------------------
def get_first_image(url):
    placeholder = "https://via.placeholder.com/150"
    if pd.isna(url):
        return placeholder

    url = str(url).strip()
    for sep in ["|", ",", " "]:
        if sep in url:
            for part in url.split(sep):
                if part.startswith("http"):
                    return part
    return url if url.startswith("http") else placeholder

# -------------------------------------------------
# DISPLAY PRODUCTS
# -------------------------------------------------
def display_products(df, cols=5):
    if df.empty:
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
            with col:
                st.markdown(f"""
                <div class="product-card">
                    <img src="{get_first_image(p['ImageURL'])}" class="product-img"/>
                    <div><b>{p['Name']}</b></div>
                    <div style="font-size:13px;color:#555;">Brand: {p['Brand']}</div>
                    <div class="badge-rating">⭐ {round(p['Rating'],2)}</div><br>
                    <div class="badge-review">📝 {p['ReviewCount']} Reviews</div>
                </div>
                """, unsafe_allow_html=True)

            idx += 1

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("🧑‍💻 User Options")
user_id = st.sidebar.number_input("User ID (0 = New User)", min_value=0)
product_name = st.sidebar.text_input("🔍 Search Product")
recommend_btn = st.sidebar.button("✨ Get Recommendations")

# -------------------------------------------------
# LOGIC
# -------------------------------------------------
if not recommend_btn:
    if user_id == 0:
        display_products(get_top_rated_items(data, 10))
    else:
        recs = collaborative_filtering_recommendations(data, user_id, 10)
        display_products(recs if not recs.empty else get_top_rated_items(data, 10))

else:
    if product_name:
        display_products(
            content_based_recommendation(data, product_name, 10)
        )
    else:
        display_products(get_top_rated_items(data, 10))

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("✨ Smart Recommendation System | Rating • Content • Collaborative • Hybrid")
