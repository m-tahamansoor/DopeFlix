import streamlit as st
import requests
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
from Model import get_hybrid_recommendations, get_hybrid_recommendations_by_keywords, prepare_data_and_models

@st.cache_resource
def load_data_and_models():
    return prepare_data_and_models()

with st.spinner("Loading data and models (this may take a few minutes)..."):
    try:
        movies, movie_text_embeddings, sbert_model, gnn_model, graph_data, node_types, original_titles = load_data_and_models()
    except Exception as e:
        st.error(f"⚠️ Error loading data and models: {str(e)}. Ensure 'credits_dataset.csv' and 'movies_dataset.csv' are in the project directory.")
        st.stop()

def fetch_poster(movie_id):
    api_key = '904d0edc5a832846953a537f96584018'
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path', '')
        return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/130"
    except:
        return "https://via.placeholder.com/130"

st.title("DopeFlix - Smart Movie Recommender")

# Initialize session state for selected movie
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

def display_movie_details(movie_id, title, overview):
    st.session_state.selected_movie = {'movie_id': movie_id, 'title': title, 'overview': overview}

tab1, tab2 = st.tabs(["Recommend by Title", "Recommend by Genre + Keywords"])

with tab1:
    title_input = st.text_input("Enter a movie title")
    if st.button("Recommend by Title"):
        if title_input:
            results, error = get_hybrid_recommendations(title_input, movies=movies, sbert_model=sbert_model, 
                                                       gnn_model=gnn_model, graph_data=graph_data, node_types=node_types,
                                                       movie_text_embeddings=movie_text_embeddings)
            if error:
                st.error(f"⚠️ {error}")
            else:
                st.subheader("Recommended Movies:")
                for i in range(0, len(results), 5):
                    cols = st.columns(5)
                    for col, j in zip(cols, range(i, min(i+5, len(results)))):
                        movie_title = original_titles[results.index[j]]  # Use original title
                        movie_id = results.iloc[j]['movie_id']
                        overview = results.iloc[j]['overview']
                        with col:
                            poster = fetch_poster(movie_id)
                            if st.image(poster, width=130, caption=movie_title, use_column_width=False):
                                display_movie_details(movie_id, movie_title, overview)
        else:
            st.error("⚠️ Please enter a movie title.")

with tab2:
    tag_input = st.text_input("Enter genre, keywords, or characters (e.g., Action war time-travel)")
    if st.button("Recommend by Genre + Keywords"):
        if tag_input:
            results, error = get_hybrid_recommendations_by_keywords(tag_input, movies=movies, sbert_model=sbert_model,
                                                                   movie_text_embeddings=movie_text_embeddings)
            if error:
                st.error(f"⚠️ {error}")
            else:
                st.subheader("Recommended Movies:")
                for i in range(0, len(results), 5):
                    cols = st.columns(5)
                    for col, j in zip(cols, range(i, min(i+5, len(results)))):
                        movie_title = original_titles[results.index[j]]  # Use original title
                        movie_id = results.iloc[j]['movie_id']
                        overview = results.iloc[j]['overview']
                        with col:
                            poster = fetch_poster(movie_id)
                            if st.image(poster, width=130, caption=movie_title, use_column_width=False):
                                display_movie_details(movie_id, movie_title, overview)
        else:
            st.error("⚠️ Please enter genre, keywords, or characters.")

# Display selected movie details
if st.session_state.selected_movie:
    st.subheader("Movie Details")
    movie = st.session_state.selected_movie
    st.write(f"**Title**: {movie['title']}")
    st.write(f"**Overview**: {movie['overview']}")
    st.image(fetch_poster(movie['movie_id']), width=300)
    if st.button("Close"):
        st.session_state.selected_movie = None