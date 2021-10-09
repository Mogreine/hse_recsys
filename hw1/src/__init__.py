import implicit
import pandas as pd
import numpy as np
import scipy.sparse as sp

from lightfm.datasets import fetch_movielens
from sklearn.metrics.pairwise import cosine_similarity

import time


class ALS:
    def __init__(self, hidden_dim: int = 10):
        self.hidden_dim = hidden_dim

    def calc_loss(self, R):
        R = R.toarray()
        non_zero_mask = R != 0
        scores = self.U @ self.I.T
        left = np.sum((R[non_zero_mask] - scores[non_zero_mask]) ** 2)

        print(left)

    def _update_matrix(self, X, Y, R, hidden_dim, l2, item: bool):
        n_objects = X.shape[0]

        # updating users
        for i in range(n_objects):
            if item:
                r = R[i].toarray().flatten()
            else:
                r = R[:, i].toarray().flatten()

            non_zero_ids_mask = r != 0
            if np.count_nonzero(non_zero_ids_mask) == 0:
                continue

            ys = Y[non_zero_ids_mask]

            # slow version of sum of outer products of y
            outer_prods = np.einsum("bi,bo->io", ys, ys)
            inv = np.eye(hidden_dim) * l2 + outer_prods

            inv = np.linalg.inv(inv)

            r_part = np.sum(r[non_zero_ids_mask].reshape(-1, 1) * ys, axis=0)

            X[i] = inv @ r_part

    def fit(self, R, hidden_dim: int = 10, n_iters: int = 10, l2: float = 0.01):
        n_users, n_items = R.shape

        U = np.random.normal(size=(n_users, hidden_dim))
        I = np.random.normal(size=(n_items, hidden_dim))

        self.U = U
        self.I = I

        for _ in range(n_iters):
            t = time.time()

            self._update_matrix(U, I, R, hidden_dim, l2, True)
            self._update_matrix(I, U, R, hidden_dim, l2, False)

            elapsed = time.time() - t
            print(f"Iteration time: {int(elapsed)}")

            self.calc_loss(R)

    def recommend(self, user_id):
        scores = self.I @ self.U[user_id]
        return reversed(np.argsort(scores))

    def similar_items(self, item_id):
        scores = cosine_similarity(self.I)
        return reversed(np.argsort(scores[item_id]))


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
    user_item = sp.coo_matrix((np.ones_like(users), (users, movies)))
    user_item_t_csr = user_item.T.tocsr()
    user_item_csr = user_item.tocsr()

    als = ALS()
    als.fit(user_item_csr, n_iters=4, hidden_dim=64)

    get_similars = lambda item_id, model: [
        movie_info[movie_info["movie_id"] == x]["name"].to_string() for x in model.similar_items(item_id)
    ]

    print(get_similars(1, als)[:10])

    get_user_history = lambda user_id, implicit_ratings: [
        movie_info[movie_info["movie_id"] == x]["name"].to_string()
        for x in implicit_ratings[implicit_ratings["user_id"] == user_id]["movie_id"]
    ]

    print(get_user_history(4, implicit_ratings)[:10])

    get_recommendations = lambda user_id, model: [
        movie_info[movie_info["movie_id"] == x]["name"].to_string() for x in model.recommend(user_id)
    ]

    print(get_recommendations(4, als)[:10])
