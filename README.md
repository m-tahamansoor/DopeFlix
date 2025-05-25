# DopeFlix - Smart Movie Recommender

DopeFlix is an intelligent movie recommendation system built using a hybrid approach that combines **Sentence-BERT (SBERT)** for text-based similarity and **Graph Convolutional Networks (GCN)** for graph-based relationships. The system provides personalized movie recommendations based on movie titles or user-defined genres, keywords, or characters. The frontend is powered by **Streamlit**, offering an interactive and user-friendly interface.

## Features
- **Title-Based Recommendations**: Enter a movie title to get similar movie recommendations based on a hybrid of text and graph embeddings.
- **Keyword-Based Recommendations**: Input genres, keywords, or actor names (e.g., "Action war Christian Bale") to receive tailored movie suggestions.
- **Interactive UI**: Built with Streamlit, the app displays movie posters, titles, and overviews, with clickable posters to view detailed movie information.
- **Hybrid Model**: Combines SBERT for semantic text similarity and GCN for capturing relationships between movies, actors, genres, keywords, directors, and production countries.
- **External Data Integration**: Fetches movie posters from The Movie Database (TMDb) API.

## Project Structure
- `Model.ipynb`: Jupyter Notebook containing the core logic for data preprocessing, SBERT embeddings, GCN model training, and recommendation functions.
- `UI.py`: Streamlit application code for the interactive frontend.
- `credits_dataset.csv`: Dataset containing movie credits (cast and crew).
- `movies_dataset.csv`: Dataset containing movie metadata (titles, genres, keywords, etc.).

## Requirements
To run the project, install the required Python packages:

```bash
pip install streamlit pandas numpy nltk sentence-transformers torch torch-geometric requests
for dataset mail me on "tahamansoor85@gmail.com"
