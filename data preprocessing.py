import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import numpy as np

# # Read only required columns with optimized types
df_ratings = pd.read_csv("data/ml-32m/ratings.csv", usecols=['movieId', 'rating'], dtype={'movieId': 'int32', 'rating': 'float32'})
# df_movies = pd.read_csv("data/ml-32m/movies.csv", usecols=['movieId', 'genres'], dtype={'movieId': 'int32'})

# # Merge on movieId
# df_genre = pd.merge(df_ratings, df_movies, on="movieId")

# # Explode genres
# df_genre['genres'] = df_genre['genres'].str.split('|')
# df_genre = df_genre.explode('genres')

# # Save result
# df_genre.to_csv("data/df_genre_ratings.csv", index=False)
# print("Saved to df_genre_ratings.csv")

# df_movie = pd.read_csv("data/ml-32m/movies.csv", usecols=['movieId', 'genres'])
# df_links = pd.read_csv("data/ml-32m/links.csv", usecols=['movieId', 'tmdbId'])
# df_tmdb = pd.read_csv("data/TMDB_movie_dataset_v11.csv", usecols=['id', 'title', 'release_date', 'runtime', 'budget', 
#                                                                   'original_language'])
# avg_ratings = (df_ratings.groupby('movieId')['rating'].mean().reset_index()).rename(columns={'rating': 'average_rating'})

# df_movie_links = df_movie.merge(df_links, on='movieId', how='inner')
# df_combined = df_movie_links.merge(df_tmdb, left_on='tmdbId', right_on='id', how='inner')

# df_combined = df_combined.drop(columns=['id', 'tmdbId'])

# df_combined['release_date'] = pd.to_datetime(df_combined['release_date'], dayfirst=True)
# df_combined['year'] = df_combined['release_date'].dt.year
# df_combined['release_date'] = df_combined['release_date'].dt.strftime('%d-%m-%Y')
# df_combined['year'] = df_combined['year'].astype('Int64').astype(str)

# df_combined = df_combined.merge(avg_ratings, on='movieId', how='left')

# df_combined.to_csv("data/df_combined.csv", index=False)
# print("Saved to df_combined.csv")


df_combined = pd.read_csv("data/df_combined.csv")

# convert genre and original language to one-hot encoding
df_combined['genres'] = df_combined['genres'].astype(str).str.split('|')
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df_combined['genres'])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=df_combined.index)
language_df = pd.get_dummies(df_combined['original_language'], prefix='lang')

# select and prepare other features
numeric_features = df_combined[['runtime', 'budget', 'average_rating', 'year']].fillna(0)

# combine features into one matrix and scale
full_feature_matrix = np.hstack([genre_df.values, language_df.values, numeric_features.values])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(full_feature_matrix)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_coords = tsne.fit_transform(scaled_features)

# add t-SNE results to df_combined
df_combined['x'] = tsne_coords[:, 0]
df_combined['y'] = tsne_coords[:, 1]

df_combined.to_csv("data/df_combined.csv", index=False)


