import implicit
import pandas as pd
import numpy as np
import scipy.sparse as sp

from typing import Tuple, List
from lightfm.datasets import fetch_movielens
from sklearn.metrics.pairwise import cosine_similarity

import time


class MatrixFactorization:

    def recommend(self, user_id):
        ...

    def similar_items(self, item_id):
        ...


class ALS(MatrixFactorization):
    def __init__(self, hidden_dim: int = 10):
        self.hidden_dim = hidden_dim

    def fit(self, R, hidden_dim: int = 10, n_iters: int = 10, l2: float = 0.01, verbose: bool = True):
        n_users, n_items = R.shape

        U = np.random.normal(size=(n_users, hidden_dim))
        I = np.random.normal(size=(n_items, hidden_dim))

        self.U = U
        self.I = I

        for step in range(n_iters):
            t = time.time()

            self._update_matrix(U, I, R, hidden_dim, l2, True)
            self._update_matrix(I, U, R, hidden_dim, l2, False)

            iteration_time = time.time() - t

            if verbose:
                mse, reg = self._calc_loss(R, l2)
                print(f"Iteration: {step + 1}; time: {iteration_time:.1f}; loss: {mse + reg:.2f}; mse: {mse:.2f}")

    def recommend(self, user_id):
        scores = self.I @ self.U[user_id]
        return reversed(np.argsort(scores))

    def similar_items(self, item_id):
        scores = cosine_similarity(self.I)
        return reversed(np.argsort(scores[item_id]))

    def _calc_loss(self, R, l2) -> Tuple[float, float]:
        R = R.toarray()
        non_zero_mask = R != 0
        scores = self.U @ self.I.T
        mse = np.sum((R[non_zero_mask] - scores[non_zero_mask]) ** 2)
        reg = l2 * (np.sum(np.linalg.norm(self.I, axis=1) ** 2)
                    + np.sum(np.linalg.norm(self.U, axis=1) ** 2))

        return mse, reg

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

            # sum of outer products of y
            outer_prods = np.einsum("bi,bo->io", ys, ys)
            inv = np.eye(hidden_dim) * l2 + outer_prods
            inv = np.linalg.inv(inv)

            # sum of r * y
            r_part = np.sum(r[non_zero_ids_mask].reshape(-1, 1) * ys, axis=0)

            X[i] = inv @ r_part


class SVD:
    def __init__(self):
        ...

    def _update_matrix(self, samples: List[Tuple[int, int, int]], U, I, lr, l2):
        shuffled_samples = np.random.permutation(samples)
        for u_id, i_id, r in shuffled_samples:
            error = U[u_id] @ I[i_id] - r
            # print(f"Error: {error}")
            U[u_id] = U[u_id] - lr * 2 * (error * I[i_id] + l2 * U[u_id])
            I[i_id] = I[i_id] - lr * 2 * (error * U[u_id] + l2 * I[i_id])

    def fit(self, R, hidden_dim: int = 10, n_iters: int = 10, l2: float = 0.01, lr: float = 0.1, verbose: bool = True):
        n_users, n_items = R.shape

        U = np.random.uniform(0, 1 / np.sqrt(hidden_dim), size=(n_users, hidden_dim))
        I = np.random.normal(0, 1 / np.sqrt(hidden_dim), size=(n_items, hidden_dim))

        self.U = U
        self.I = I

        samples = [(i, j, k) for i, j, k in zip(R.row, R.col, R.data)]

        for step in range(n_iters):
            t = time.time()

            self._update_matrix(samples, U, I, lr, l2)

            iteration_time = time.time() - t

            if verbose:
                mse, reg = self._calc_loss(R, l2)
                print(f"Iteration: {step + 1}; time: {iteration_time:.1f}; loss: {mse + reg:.2f}; mse: {mse:.2f}")

    def _calc_loss(self, R, l2) -> Tuple[float, float]:
        R = R.toarray()
        non_zero_mask = R != 0
        scores = self.U @ self.I.T
        mse = np.sum((R[non_zero_mask] - scores[non_zero_mask]) ** 2)
        reg = l2 * (np.sum(np.linalg.norm(self.I, axis=1) ** 2)
                    + np.sum(np.linalg.norm(self.U, axis=1) ** 2))

        return mse, reg


class BPR(MatrixFactorization):
    def __init__(self):
        ...

