import implicit
import pandas as pd
import numpy as np
import scipy.sparse as sp

from typing import Tuple
from lightfm.datasets import fetch_movielens
from sklearn.metrics.pairwise import cosine_similarity

import time
from hw1.src.methods import BPR, ALS, SVD
from hw1.src.utils import calc_auc, test_model


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

    model = SVD(lr=0.01, l2=0.01)
    model.fit(user_item_exp, n_iters=2, hidden_dim=64)

    # model = BPR(lr=1e-2, l2=0.01)
    # model.fit(user_item_csr, n_iters=8000, hidden_dim=64, batch_size=user_item_csr.shape[0])
    # print(f"AUC: {calc_auc(model, user_item_csr)}")
    # print(f"AUC: {roc_auc(model, user_item_csr)}")

    # model = ALS(l2=1)
    # model.fit(user_item_csr, n_iters=5, hidden_dim=64)

    test_model(model, implicit_ratings, movie_info)
