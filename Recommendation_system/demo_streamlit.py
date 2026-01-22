import streamlit as st
import pandas as pd

from preprocess_data import process_data
from rating_based_recommendation import get_top_rated_items
from content_based_filtering import content_based_recommendation
from collaborative_based_filtering import collaborative_filtering_recommendations

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Recommendation System",
    layout="wide"
)

st.title("🌈 Smart Product Recommendation System")

# -------------------------------------------------
# COLORFUL & ATTRACTIVE CSS
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
    transition: all 0.3s ease;
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

.product-name {
    font-weight: 600;
    font-size: 15px;
    margin: 8px 0;
    color: #222;
}

.product-meta {
    font-size: 13px;
    color: #555;
}

.rating-badge {
    display: inline-block;
    background: linear-gradient(135deg, #00c853, #64dd17);
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    margin-top: 6px;
}

.review-badge {
    display: inline-block;
    background: linear-gradient(135deg, #2962ff, #448aff);
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    margin-top: 6px;
}

h2, h3 {
    color: #ff6a00;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load & preprocess data
# -------------------------------------------------
@st.cache_data
# def load_data():
#     raw = pd.read_csv("clean_data.csv")
#     return process_data(raw)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, "clean_data.csv")
    raw = pd.read_csv(data_path)
    return raw


data = load_data()

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def get_first_image(url):
    """
    Always return a valid image URL.
    Uses placeholder if URL is broken.
    """
    placeholder = "https://via.placeholder.com/150"

    if pd.isna(url) or str(url).strip() == "":
        return placeholder

    url = str(url).strip()

    # Take only first URL if multiple exist
    for sep in ["|", ",", " "]:
        if sep in url:
            url = url.split(sep)[0].strip()

    if not url.startswith("http"):
        return placeholder

    return url

def find_matching_product(data, user_input):
    matches = data[data["Name"].str.contains(
        user_input,
        case=False,
        na=False,
        regex=False   # 🔥 prevents regex crash
    )]
    if matches.empty:
        return None
    return matches.iloc[0]["Name"]

def get_multi_product_recommendations(data, user_input, top_n=5):
    products = [p.strip() for p in user_input.split(",") if p.strip()]
    all_recs = []

    for prod in products:
        matched = find_matching_product(data, prod)
        if matched:
            recs = content_based_recommendation(
                data,
                item_name=matched,
                top_n=top_n
            )
            all_recs.append(recs)

    if not all_recs:
        return pd.DataFrame()

    return pd.concat(all_recs).drop_duplicates().head(top_n * 2)

# -------------------------------------------------
# Display products safely
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

            name = p.get("Name", "N/A")
            brand = p.get("Brand", "N/A")
            rating = p.get("Rating", "N/A")
            reviews = p.get("ReviewCount", "N/A")
            img = get_first_image(p.get("ImageURL", ""))

            with col:
                st.markdown(f"""
                <div class="product-card">
                    <img src="{img}" class="product-img"/>
                    <div class="product-name">{name}</div>
                    <div class="product-meta">Brand: {brand}</div>
                    <div class="rating-badge">⭐ {rating}</div><br>
                    <div class="review-badge">📝 {reviews} Reviews</div>
                </div>
                """, unsafe_allow_html=True)

            idx += 1

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("🧑‍💻 User Options")

user_id = st.sidebar.number_input(
    "User ID (0 = New User)",
    min_value=0,
    step=1
)

product_name = st.sidebar.text_input(
    "🔍 Search products (comma separated allowed)"
)

recommend_btn = st.sidebar.button("✨ Get Recommendations")

# -------------------------------------------------
# Recommendation Logic
# -------------------------------------------------
if recommend_btn:

    # 🔍 SEARCH FLOW
    if product_name.strip():
        st.subheader("🔍 Similar Products")

        search_recs = get_multi_product_recommendations(
            data,
            product_name,
            top_n=5
        )
        display_products(search_recs)

        # ✨ You may also like
        if user_id > 0:
            st.markdown("---")
            st.subheader("✨ You may also like")

            collab_recs = collaborative_filtering_recommendations(
                data,
                target_user_id=user_id,
                top_n=5
            )

            if not search_recs.empty and "Name" in collab_recs.columns:
                collab_recs = collab_recs[
                    ~collab_recs["Name"].isin(search_recs["Name"])
                ]

            display_products(collab_recs)

    # ⭐ NEW USER
    elif user_id == 0:
        st.subheader("⭐ Top Rated Products")
        display_products(get_top_rated_items(data, top_n=10))

    # 🎯 EXISTING USER
    else:
        st.subheader("🎯 Personalized Recommendations")

        user_history = data[data["ID"] == user_id]
        if user_history.empty:
            display_products(get_top_rated_items(data, top_n=10))
        else:
            last_item = user_history.iloc[-1]["Name"]
            st.caption(f"Because you liked **{last_item}**")

            content_rec = content_based_recommendation(
                data,
                item_name=last_item,
                top_n=5
            )

            collab_rec = collaborative_filtering_recommendations(
                data,
                target_user_id=user_id,
                top_n=5
            )

            final_rec = (
                pd.concat([content_rec, collab_rec])
                .drop_duplicates()
                .head(10)
            )

            display_products(final_rec)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("✨ Colorful Recommendation System • Content • Collaborative • Hybrid")
