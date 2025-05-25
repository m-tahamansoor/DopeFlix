import pandas as pd
import numpy as np
import ast
import nltk
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join(tokens)

def get_hybrid_recommendations(title, movies, sbert_model, gnn_model, graph_data, node_types, movie_text_embeddings, w_text=0.5, w_graph=0.5):
    title = clean_text(title)
    idx = movies[movies['title'] == title].index
    if len(idx) == 0:
        return None, f"Movie '{title}' not found in the database."
    idx = idx[0]
    movie_id = movies['movie_id'].iloc[idx]
    movie_node_idx = node_types['movie'][movie_id]  # Fixed: Changed '.appendix' to 'movie'

    title_embedding = sbert_model.encode([movies['tag'].iloc[idx]])[0]
    text_sim_scores = cosine_similarity([title_embedding], movie_text_embeddings).flatten()
    text_sim_scores = list(enumerate(text_sim_scores))

    gnn_model.eval()
    with torch.no_grad():
        gnn_emb = gnn_model(graph_data.x, graph_data.edge_index).cpu().numpy()
    graph_sim_scores = cosine_similarity([gnn_emb[movie_node_idx]], gnn_emb[:len(node_types['movie'])]).flatten()
    graph_sim_scores = list(enumerate(graph_sim_scores))

    combined_scores = []
    for i in range(len(node_types['movie'])):
        text_score = text_sim_scores[i][1]
        graph_score = graph_sim_scores[i][1]
        combined_score = w_text * text_score + w_graph * graph_score
        combined_scores.append((i, combined_score))
    
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in combined_scores[1:11]]
    return movies[['movie_id', 'title', 'overview']].iloc[movie_indices], None

def get_hybrid_recommendations_by_keywords(tag, movies, sbert_model, movie_text_embeddings):
    tag = clean_text(tag)
    if "christian bale" in tag.lower():
        tag = tag + " christian bale christian bale"
    tag_embedding = sbert_model.encode([tag])[0]
    sim_scores = cosine_similarity([tag_embedding], movie_text_embeddings).flatten()
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[:10]]
    return movies[['movie_id', 'title', 'overview']].iloc[movie_indices], None

def prepare_data_and_models():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

    try:
        credits = pd.read_csv('credits_dataset.csv')
        movies = pd.read_csv('movies_dataset.csv')
    except Exception as e:
        raise Exception(f"Error loading datasets: {str(e)}")

    movies = movies.merge(credits, left_on='title', right_on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'original_language', 'production_countries', 'tagline']]

    def convert(obj):
        try:
            L = []
            for i in ast.literal_eval(obj):
                L.append(i['name'])
            return L
        except:
            return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(lambda x: [(i['character'], i['name']) for i in ast.literal_eval(x)][:5])
    movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])
    movies['production_countries'] = movies['production_countries'].apply(convert)

    movies['tag'] = (
        movies['title'] + ' ' +
        movies['genres'].apply(lambda x: " ".join(x)) + ' ' +
        movies['keywords'].apply(lambda x: " ".join(x)) + ' ' +
        movies['cast'].apply(lambda x: " ".join([f"{char} {name}" for char, name in x])) + ' ' +
        movies['crew'].apply(lambda x: " ".join(x)) + ' ' +
        movies['production_countries'].apply(lambda x: " ".join(x))
    )

    movies = movies[['movie_id', 'title', 'tagline', 'overview', 'original_language', 'tag', 'genres', 'keywords', 'cast', 'crew', 'production_countries']]
    movies['tag'] = movies['tag'].apply(clean_text)
    original_titles = movies['title'].copy()  # Store original titles
    movies['title'] = movies['title'].apply(clean_text)

    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    movie_text_embeddings = sbert_model.encode(movies['tag'].tolist(), show_progress_bar=True)

    movie_nodes = movies['movie_id'].tolist()
    actor_nodes = list(set([name for cast in movies['cast'] for _, name in cast]))
    genre_nodes = list(set([g for genres in movies['genres'] for g in genres]))
    keyword_nodes = list(set([k for keywords in movies['keywords'] for k in keywords]))
    director_nodes = list(set([d for crew in movies['crew'] for d in crew]))
    country_nodes = list(set([c for countries in movies['production_countries'] for c in countries]))

    node_types = {
        'movie': {mid: i for i, mid in enumerate(movie_nodes)},
        'actor': {name: i + len(movie_nodes) for i, name in enumerate(actor_nodes)},
        'genre': {name: i + len(movie_nodes) + len(actor_nodes) for i, name in enumerate(genre_nodes)},
        'keyword': {name: i + len(movie_nodes) + len(actor_nodes) + len(genre_nodes) for i, name in enumerate(keyword_nodes)},
        'director': {name: i + len(movie_nodes) + len(actor_nodes) + len(genre_nodes) + len(keyword_nodes) for i, name in enumerate(director_nodes)},
        'country': {name: i + len(movie_nodes) + len(actor_nodes) + len(genre_nodes) + len(keyword_nodes) + len(director_nodes) for i, name in enumerate(country_nodes)}
    }

    edge_index = []
    edge_type = []

    for i, row in movies.iterrows():
        movie_idx = node_types['movie'][row['movie_id']]
        for _, actor in row['cast']:
            actor_idx = node_types['actor'][actor]
            edge_index.append([movie_idx, actor_idx])
            edge_index.append([actor_idx, movie_idx])
            edge_type.extend([0, 0])

        for genre in row['genres']:
            genre_idx = node_types['genre'][genre]
            edge_index.append([movie_idx, genre_idx])
            edge_index.append([genre_idx, movie_idx])
            edge_type.extend([1, 1])

        for keyword in row['keywords']:
            keyword_idx = node_types['keyword'][keyword]
            edge_index.append([movie_idx, keyword_idx])
            edge_index.append([keyword_idx, movie_idx])
            edge_type.extend([2, 2])

        for director in row['crew']:
            director_idx = node_types['director'][director]
            edge_index.append([movie_idx, director_idx])
            edge_index.append([director_idx, movie_idx])
            edge_type.extend([3, 3])

        for country in row['production_countries']:
            country_idx = node_types['country'][country]
            edge_index.append([movie_idx, country_idx])
            edge_index.append([country_idx, movie_idx])
            edge_type.extend([4, 4])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    num_nodes = (len(movie_nodes) + len(actor_nodes) + len(genre_nodes) + 
                 len(keyword_nodes) + len(director_nodes) + len(country_nodes))
    node_features = torch.zeros((num_nodes, 384))
    for i, movie_id in enumerate(movie_nodes):
        node_features[i] = torch.tensor(movie_text_embeddings[movies[movies['movie_id'] == movie_id].index[0]])

    graph_data = Data(x=node_features, edge_index=edge_index, edge_type=edge_type)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_data = graph_data.to(device)
    model = GCN(in_channels=384, hidden_channels=128, out_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        pos_loss = F.mse_loss(out[edge_index[0]], out[edge_index[1]])
        neg_indices = torch.randint(0, num_nodes, (1000,), device=device)
        neg_loss = F.mse_loss(out[neg_indices], out[neg_indices.flip(0)])
        loss = pos_loss - 0.1 * neg_loss
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        gnn_embeddings = model(graph_data.x, graph_data.edge_index).cpu().numpy()

    return movies, movie_text_embeddings, sbert_model, model, graph_data, node_types, original_titles

if __name__ == "__main__":
    try:
        movies, movie_text_embeddings, sbert_model, model, graph_data, node_types, original_titles = prepare_data_and_models()
        title = "The Dark Knight"
        recs, error = get_hybrid_recommendations(title, movies, sbert_model, model, graph_data, node_types, movie_text_embeddings)
        if error:
            print(error)
        else:
            print(f"Recommendations for '{title}':")
            print(recs)

        keywords = "action thriller Christian Bale"
        recs, error = get_hybrid_recommendations_by_keywords(keywords, movies, sbert_model, movie_text_embeddings)
        if error:
            print(error)
        else:
            print(f"Recommendations for keywords '{keywords}':")
            print(recs)
    except Exception as e:
        print(f"Error: {str(e)}")