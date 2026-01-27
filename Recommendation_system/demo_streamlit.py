import streamlit as st
import pandas as pd
import os

from preprocess_data import process_data
from rating_based_recommendation import get_top_rated_items
from content_based_filtering import content_based_recommendation
from collaborative_based_filtering import collaborative_filtering_recommendations

# -------------------------------------------------
# Page Config (ONLY ONCE, FIRST)
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Recommendation System",
    layout="wide"
)

st.title("🛍️ Smart Product Recommendation System")

st.set_page_config(
    page_title="Smart Recommendation System",
    layout="wide"
)

st.title("My Smart Product Recommendation System")

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
# File Paths (CLOUD SAFE)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "clean_data.csv")

# -------------------------------------------------
# Load & preprocess data
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "clean_data.csv")

@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_PATH)
    data = process_data(raw)

    # ---- NORMALIZE COLUMN NAMES ----
    data.columns = [c.strip() for c in data.columns]

    # ---- FORCE RATING CLEANING ----
    if "Rating" in data.columns:
        data["Rating"] = pd.to_numeric(data["Rating"], errors="coerce").fillna(0).round(2)

    # ---- FORCE IMAGE CLEANING ----
    if "ImageURL" in data.columns:
        data["ImageURL"] = data["ImageURL"].astype(str)

    return data

data = load_data()

# -------------------------------------------------
# SMART IMAGE HANDLER (FIXED)
# -------------------------------------------------
def get_first_image(url):
    placeholder = "https://via.placeholder.com/150"

    if pd.isna(url):
        return placeholder

    url = str(url).strip()

    if url == "" or url.lower() == "nan":
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

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def find_matching_product(data, user_input):
    matches = data[data["Name"].str.contains(user_input, case=False, na=False)]
    if matches.empty:
        return None
    return matches.iloc[0]["Name"]

def get_multi_product_recommendations(data, user_input, top_n=5):
    products = [p.strip() for p in user_input.split(",") if p.strip()]
    all_recs = []

    for prod in products:
        matched = find_matching_product(data, prod)
        if matched:
            recs = content_based_recommendation(data, matched, top_n)
            all_recs.append(recs)

    if not all_recs:
        return pd.DataFrame()

    return pd.concat(all_recs).drop_duplicates().head(top_n * 2)

# -------------------------------------------------
# Display products (FIXED)
# -------------------------------------------------
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

            name = str(p.get("Name", "N/A"))
            brand = str(p.get("Brand", "N/A"))
            rating = p.get("Rating", 0)
            reviews = p.get("ReviewCount", 0)
            img = get_first_image(p.get("ImageURL", ""))

            try:
                rating = round(float(rating), 2)
            except:
                rating = 0.0

            with col:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(145deg, #ffffff, #f3f7ff);
                    border-radius: 18px;
                    padding: 16px;
                    text-align: center;
                    box-shadow: 0 8px 22px rgba(0,0,0,0.12);
                ">
                    <img src="{img}" style="
                        width:140px;height:140px;border-radius:50%;
                        object-fit:cover;border:4px solid #ff7a00;
                        margin-bottom:12px;
                    "/>
                    <div style="font-weight:600;font-size:15px;">{name}</div>
                    <div style="font-size:13px;color:#555;">Brand: {brand}</div>
                    <div style="background:#00c853;color:white;padding:4px 10px;
                                border-radius:12px;font-size:12px;margin-top:6px;">
                        ⭐ {rating}
                    </div><br>
                    <div style="background:#2962ff;color:white;padding:4px 10px;
                                border-radius:12px;font-size:12px;">
                        📝 {reviews} Reviews
                    </div>
                </div>
                """, unsafe_allow_html=True)

            idx += 1

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("🧑‍💻 User Options")

user_id = st.sidebar.number_input("User ID (0 = New User)", min_value=0, step=1)

product_name = st.sidebar.text_input("🔍 Search products (comma separated allowed)")

recommend_btn = st.sidebar.button("✨ Get Recommendations")

# -------------------------------------------------
# HOME PAGE DEFAULT RECOMMENDATIONS
# -------------------------------------------------
if not recommend_btn:

    if user_id == 0:
        st.subheader("⭐ Top Rated Products (Trending Now)")
        display_products(get_top_rated_items(data, 10))

    else:
        st.subheader("🎯 Personalized Recommendations For You")

        collab_recs = collaborative_filtering_recommendations(
            data, target_user_id=user_id, top_n=10
        )

        if collab_recs.empty:
            st.info("Not enough data — showing trending products instead.")
            display_products(get_top_rated_items(data, 10))
        else:
            display_products(collab_recs)

# -------------------------------------------------
# ADVANCED RECOMMENDATION FLOW
# -------------------------------------------------
if recommend_btn:

    if product_name.strip():
        st.subheader("🔍 Similar Products (Content-Based)")

        search_recs = get_multi_product_recommendations(data, product_name, 5)
        display_products(search_recs)

        if user_id > 0:
            st.markdown("---")
            st.subheader("✨ You may also like (Collaborative)")

            collab_recs = collaborative_filtering_recommendations(
                data, target_user_id=user_id, top_n=5
            )

            display_products(collab_recs)

    elif user_id == 0:
        st.subheader("⭐ Top Rated Products")
        display_products(get_top_rated_items(data, 10))

    else:
        st.subheader("🎯 Hybrid Personalized Recommendations")

        user_history = data[data["ID"] == user_id]

        if user_history.empty:
            display_products(get_top_rated_items(data, 10))
        else:
            last_item = user_history.iloc[-1]["Name"]
            st.caption(f"Because you liked **{last_item}**")

            content_rec = content_based_recommendation(data, last_item, 5)
            collab_rec = collaborative_filtering_recommendations(data, user_id, 5)

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
st.caption("✨ Smart Recommendation System | Rating • Content • Collaborative • Hybrid")
