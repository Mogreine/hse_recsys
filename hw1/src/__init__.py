import implicit
import pandas as pd
import numpy as np
import scipy.sparse as sp

from typing import Tuple
from lightfm.datasets import fetch_movielens
from sklearn.metrics.pairwise import cosine_similarity

import time
from hw1.src.methods import *


if __name__ == "__main__":
    ratings = pd.read_csv(
        "../data/ml-1m/ratings.dat",
        delimiter="::",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        usecols=["user_id", "movie_id", "rating"],
        engine="python",
    )

    movie_info = pd.read_csv(
        "../data/ml-1m/movies.dat",
        delimiter="::",
        header=None,
        names=["movie_id", "name", "category"],
        engine="python",
    )

    implicit_ratings = ratings.loc[(ratings["rating"] >= 4)]

    users = implicit_ratings["user_id"]
    movies = implicit_ratings["movie_id"]
    user_item_exp = sp.coo_matrix((ratings["rating"], (ratings["user_id"], ratings["movie_id"])))
    user_item = sp.coo_matrix((np.ones_like(users), (users, movies)))
    user_item_t_csr = user_item.T.tocsr()
    user_item_csr = user_item.tocsr()

    # svd = SVD(lr=0.01)
    # svd.fit(user_item_exp, n_iters=1000, hidden_dim=64)

    # model = BPR(l2=0.01)
    # model.fit(user_item_csr, n_iters=500, hidden_dim=64, batch_size=1000)

    model = ALS(l2=1)
    model.fit(user_item_csr, n_iters=5, hidden_dim=64)

    get_similars = lambda item_id, model: [
        movie_info[movie_info["movie_id"] == x]["name"].to_string() for x in model.similar_items(item_id)
    ]

    print(get_similars(1, model)[:10])

    get_user_history = lambda user_id, implicit_ratings: [
        movie_info[movie_info["movie_id"] == x]["name"].to_string()
        for x in implicit_ratings[implicit_ratings["user_id"] == user_id]["movie_id"]
    ]

    print(get_user_history(4, implicit_ratings)[:10])

    get_recommendations = lambda user_id, model: [
        movie_info[movie_info["movie_id"] == x]["name"].to_string() for x in model.recommend(user_id)
    ]

    print(get_recommendations(4, model)[:10])
