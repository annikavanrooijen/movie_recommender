import pandas as pd
import pickle
import streamlit as st


movies = pd.read_csv("./data/movies.csv")

with open('model_neighbor.pkl', 'rb') as file:
    model = pickle.load(file)


def recommend_neighborhood(query, model, movies, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """
    matrix = pd.read_csv("./data/user_item_matrix.csv", index_col=0)

    # generate a new user from the input query of the function
    new_user = pd.DataFrame(query, index = ["new_user"], columns= matrix.columns)
    
    # filling the missing values with 0
    new_user  = new_user.fillna(0)
    
    # calculates the distances of the input data to all other users in the data
    _, neighbor_ids = model.kneighbors(
    new_user,
    n_neighbors=5,
    return_distance=True
    )

    query_ids = {}
    for title, rating in query.items():
        movie_id = movies[movies["title"] == title]["movie_id"].iloc[0]
        query_ids[str(movie_id)] = rating

    # Filtering similar users
    neighborhood = matrix.iloc[neighbor_ids[0]]
    
    #filter out movies allready seen by the user
    neighborhood_filtered = neighborhood.drop(query_ids.keys(), axis=1)
    df_score = neighborhood_filtered.sum()
    
    # ranking the top 5 movies
    df_score_ranked = df_score.sort_values(ascending = False).index.tolist()
    recommendations = df_score_ranked[:k]
    recommendation = movies.iloc[recommendations]
    
    return recommendation


