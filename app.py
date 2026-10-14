import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- 1. PAGE CONFIG & AESTHETICS ---
st.set_page_config(page_title="R@mij@", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #e04343;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    # Ensure movies.csv is in the same folder
    df = pd.read_csv('movies.csv')
    df['overview'] = df['overview'].fillna('')
    return df

try:
    df = load_data()
    
    # --- 3. ML LOGIC (TF-IDF & COSINE SIMILARITY) ---
    # This part converts text descriptions into math vectors
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    
    # Calculating the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # --- 4. USER INTERFACE ---
    st.title("ML Movie Recommendation System")
    st.subheader("Developed for Practical Knowledge")
    st.info("This system uses Natural Language Processing (NLP) to find similarities between movie plots.")

    # Sidebar for project info
    st.sidebar.header("Project Details")
    st.sidebar.write("**Algorithm:** Content-Based Filtering")
    st.sidebar.write("**Metric:** Cosine Similarity")
    st.sidebar.write("**Dataset:** TMDB 5000 Movies")

    # Dropdown for movie selection
    movie_list = df['original_title'].values
    selected_movie = st.selectbox("?? Search or Select a Movie:", movie_list)

    if st.button('Get Recommendations'):
        # Get index of the selected movie
        idx = df.index[df['original_title'] == selected_movie][0]
        
        # Get pairwise similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        
        st.write(f"### Because you liked **{selected_movie}**, you might also enjoy:")
        
        # Display recommendations in 5 columns
        cols = st.columns(5)
        for i, (m_idx, score) in enumerate(sim_scores):
            with cols[i]:
                st.success(df.iloc[m_idx]['original_title'])
                st.metric("Match Score", f"{round(score*100, 1)}%")

except FileNotFoundError:
    st.error("Error: 'movies.csv' not found. Please ensure the dataset is in the project folder.")