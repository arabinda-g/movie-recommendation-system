from surprise import KNNBasic, Dataset, Reader
import pandas as pd

# Load the data
ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
movies_columns = ['movie_id', 'title'] # Add other columns as needed
ratings = pd.read_csv('u.data', sep='\t', names=ratings_columns, encoding='latin-1')
movies = pd.read_csv('u.item', sep='|', names=movies_columns, encoding='latin-1', usecols=[0, 1])

# Reader and data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)

# Train an item-based collaborative filtering model
sim_options = {
    'name': 'cosine',
    'user_based': False  # Compute similarities between items
}
model = KNNBasic(sim_options=sim_options)
trainingSet = data.build_full_trainset()
model.fit(trainingSet)


def find_movies_by_title(search_term):
    # Filter movies that contain the search term in their title
    search_term = search_term.lower()
    matched_movies = movies[movies['title'].str.lower().str.contains(search_term)]
    return matched_movies

def get_similar_movies(movie_id, k):
    try:
        # Find similar items
        movie_inner_id = model.trainset.to_inner_iid(movie_id)
        movie_neighbors = model.get_neighbors(movie_inner_id, k=k)

        # Convert inner ids of the neighbors into names
        movie_neighbors = (model.trainset.to_raw_iid(inner_id) for inner_id in movie_neighbors)
        movie_neighbors = (movies[movies['movie_id'] == id]['title'].iloc[0] for id in movie_neighbors)

        return list(movie_neighbors)
    except ValueError:
        # This happens if the movie is not in the training set
        return []

# Example usage
search_term = "iron"
matched_movies = find_movies_by_title(search_term)

# For each matched movie, find similar movies
for _, row in matched_movies.iterrows():
    print(f"Movies similar to {row['title']}:")
    similar_movies = get_similar_movies(row['movie_id'], 10)
    for movie in similar_movies:
        print(f"    {movie}")
    print("\n")
