import time
import implicit
import pandas as pd
import numpy as np
import scipy.sparse as sp

from typing import Tuple, List
from lightfm.datasets import fetch_movielens
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import expit


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        return time.time() - start

    return wrapper


class MatrixFactorization:
    def __init__(self, lr: float = 0.01, l2: float = 0.01):
        self.lr = lr
        self.l2 = l2

        self.I = np.ndarray
        self.U = np.ndarray

    def _calc_loss(self, R) -> Tuple[float, float, float]:
        R = R.toarray()
        non_zero_mask = R != 0
        scores = self.U @ self.I.T
        mse = np.sum((R[non_zero_mask] - scores[non_zero_mask]) ** 2)
        rmse = np.sqrt(mse / np.count_nonzero(non_zero_mask))
        reg = self.l2 * (np.sum(np.linalg.norm(self.I, axis=1) ** 2) + np.sum(np.linalg.norm(self.U, axis=1) ** 2))

        return mse, reg, rmse

    def _print_info(self, R, step, iteration_time, verbose):
        if verbose:
            mse, reg, rmse = self._calc_loss(R)
            print(f"Iteration: {step + 1}; time: {iteration_time:.1f}; loss: {mse + reg:.2f}; mse: {mse:.2f}; rmse: {rmse: .5f}")

    def _initialize_matrix(self, n_users, n_items, hidden_dim):
        self.U = np.random.uniform(0, 1 / np.sqrt(hidden_dim), size=(n_users, hidden_dim))
        self.I = np.random.normal(0, 1 / np.sqrt(hidden_dim), size=(n_items, hidden_dim))

    def recommend(self, user_id):
        scores = self.I @ self.U[user_id]
        return reversed(np.argsort(scores))

    def similar_items(self, item_id):
        scores = cosine_similarity(self.I)
        return reversed(np.argsort(scores[item_id]))


class ALS(MatrixFactorization):
    def fit(self, R, hidden_dim: int = 10, n_iters: int = 10, verbose: bool = True):
        n_users, n_items = R.shape
        self._initialize_matrix(n_users, n_items, hidden_dim)

        for step in range(n_iters):
            iteration_time = self._update_matrix(self.U, self.I, R, hidden_dim, True) + self._update_matrix(self.I, self.U, R, hidden_dim, False)

            self._print_info(R, step, iteration_time, verbose)

    @time_it
    def _update_matrix(self, X, Y, R, hidden_dim, item: bool):
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
            outer_prods = Y.T @ Y
            inv = np.eye(hidden_dim) * self.l2 + outer_prods
            inv = np.linalg.inv(inv)

            # sum of r * y
            r_part = np.sum(r[non_zero_ids_mask].reshape(-1, 1) * ys, axis=0)

            X[i] = inv @ r_part


class SVD(MatrixFactorization):
    def fit(self, R, hidden_dim: int = 10, n_iters: int = 10, verbose: bool = True):
        n_users, n_items = R.shape
        self._initialize_matrix(n_users, n_items, hidden_dim)

        samples = [(i, j, k) for i, j, k in zip(R.row, R.col, R.data)]

        for step in range(n_iters):
            iteration_time = self._update_matrix(samples)

            self._print_info(R, step, iteration_time, verbose)

    @time_it
    def _update_matrix(self, samples: List[Tuple[int, int, int]]):
        U, I = self.U, self.I
        lr, l2 = self.lr, self.l2

        shuffled_samples = np.random.permutation(samples)
        for u_id, i_id, r in shuffled_samples:
            error = U[u_id] @ I[i_id] - r
            U[u_id] = U[u_id] - lr * 2 * (error * I[i_id] + l2 * U[u_id])
            I[i_id] = I[i_id] - lr * 2 * (error * U[u_id] + l2 * I[i_id])


class BPR(MatrixFactorization):
    def fit(self, R, hidden_dim: int = 10, n_iters: int = 10, batch_size: int = 32, verbose: bool = True):
        n_users, n_items = R.shape
        self._initialize_matrix(n_users, n_items, hidden_dim)

        indptr = R.indptr
        indices = R.indices
        n_users, n_items = R.shape

        for step in range(n_iters):
            users, items_pos, items_neg = self._sample(n_users, n_items, batch_size, indices, indptr)
            iteration_time = self._grad_step(users, items_pos, items_neg)

            self._print_info(R, step, iteration_time, verbose)

    @time_it
    def _grad_step(self, u, ii, ij):
        user_u = self.U[u]
        item_i = self.I[ii]
        item_j = self.I[ij]

        r_uij = np.sum(user_u * (item_i - item_j), axis=1)
        sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))

        # repeat the 1 dimension sigmoid n_factors times so
        # the dimension will match when doing the update
        hidden_dim = self.I.shape[1]
        sigmoid_tiled = np.tile(sigmoid, (hidden_dim, 1)).T

        # update using gradient descent
        grad_u = sigmoid_tiled * (item_j - item_i) + self.l2 * user_u
        grad_i = sigmoid_tiled * -user_u + self.l2 * item_i
        grad_j = sigmoid_tiled * user_u + self.l2 * item_j
        self.U[u] -= self.lr * grad_u
        self.I[ii] -= self.lr * grad_i
        self.I[ij] -= self.lr * grad_j

    def _sample(self, n_users, n_items, batch_size, indices, indptr):
        """sample batches of random triplets u, i, j"""
        sampled_pos_items = np.zeros(batch_size, dtype=np.int)
        sampled_neg_items = np.zeros(batch_size, dtype=np.int)
        sampled_users = np.random.choice(n_users, size=batch_size, replace=False)

        for idx, user in enumerate(sampled_users):
            pos_items = indices[indptr[user] : indptr[user + 1]]

            if len(pos_items) == 0:
                continue

            pos_item = np.random.choice(pos_items)
            neg_item = np.random.choice(n_items)
            while neg_item in pos_items:
                neg_item = np.random.choice(n_items)

            sampled_pos_items[idx] = pos_item
            sampled_neg_items[idx] = neg_item

        return sampled_users, sampled_pos_items, sampled_neg_items
