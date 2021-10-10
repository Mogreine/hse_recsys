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

    def _calc_loss(self, R) -> Tuple[float, float]:
        R = R.toarray()
        non_zero_mask = R != 0
        scores = self.U @ self.I.T
        mse = np.sum((R[non_zero_mask] - scores[non_zero_mask]) ** 2)
        reg = self.l2 * (np.sum(np.linalg.norm(self.I, axis=1) ** 2)
                    + np.sum(np.linalg.norm(self.U, axis=1) ** 2))

        return mse, reg

    def _initialize_matrix(self, n_users, n_items, hidden_dim):
        self.U = np.random.uniform(0, 1 / np.sqrt(hidden_dim), size=(n_users, hidden_dim))
        self.I = np.random.normal(0, 1 / np.sqrt(hidden_dim), size=(n_items, hidden_dim))

    def recommend(self, user_id):
        scores = self.I @ self.U[user_id]
        return reversed(np.argsort(scores) + 1)

    def similar_items(self, item_id):
        scores = cosine_similarity(self.I)
        return reversed(np.argsort(scores[item_id]) + 1)


class ALS(MatrixFactorization):
    def fit(self, R, hidden_dim: int = 10, n_iters: int = 10, verbose: bool = True):
        n_users, n_items = R.shape
        self._initialize_matrix(n_users, n_items, hidden_dim)

        for step in range(n_iters):
            t = time.time()

            self._update_matrix(self.U, self.I, R, hidden_dim, True)
            self._update_matrix(self.I, self.U, R, hidden_dim, False)

            iteration_time = time.time() - t

            if verbose:
                mse, reg = self._calc_loss(R)
                print(f"Iteration: {step + 1}; time: {iteration_time:.1f}; loss: {mse + reg:.2f}; mse: {mse:.2f}")

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
            outer_prods = np.einsum("bi,bo->io", ys, ys)
            inv = np.eye(hidden_dim) * self.l2 + outer_prods
            inv = np.linalg.inv(inv)

            # sum of r * y
            r_part = np.sum(r[non_zero_ids_mask].reshape(-1, 1) * ys, axis=0)

            X[i] = inv @ r_part


class SVD(MatrixFactorization):
    def _update_matrix(self, samples: List[Tuple[int, int, int]]):
        U, I = self.U, self.I
        lr, l2 = self.lr, self.l2

        shuffled_samples = np.random.permutation(samples)
        for u_id, i_id, r in shuffled_samples:
            error = U[u_id] @ I[i_id] - r
            U[u_id] = U[u_id] - lr * 2 * (error * I[i_id] + l2 * U[u_id])
            I[i_id] = I[i_id] - lr * 2 * (error * U[u_id] + l2 * I[i_id])

    def fit(self, R, hidden_dim: int = 10, n_iters: int = 10, verbose: bool = True):
        n_users, n_items = R.shape
        self._initialize_matrix(n_users, n_items, hidden_dim)

        samples = [(i, j, k) for i, j, k in zip(R.row, R.col, R.data)]

        for step in range(n_iters):
            t = time.time()

            self._update_matrix(samples)

            iteration_time = time.time() - t

            if verbose:
                mse, reg = self._calc_loss(R)
                print(f"Iteration: {step + 1}; time: {iteration_time:.1f}; loss: {mse + reg:.2f}; mse: {mse:.2f}")


class BPR(MatrixFactorization):
    @time_it
    def _grad_step(self, u, ii, ij):
        U = self.U
        I = self.I

        # u = np.random.randint(0, len(self.U))
        # ii = np.random.randint(0, len(self.I))
        # ij = np.random.randint(0, len(self.I))

        u_grad = (I[ij] - I[ii]) * expit(U[u] @ I[ij] - U[u] @ I[ii]) + self.l2 * U[u]
        ii_grad = -U[u] * expit(U[u] @ I[ij] - U[u] @ I[ii]) + self.l2 * I[ii]
        ij_grad = U[u] * expit(U[u] @ I[ij] - U[u] @ I[ii]) + self.l2 * I[ij]

        U[u] -= self.lr * u_grad
        I[ii] -= self.lr * ii_grad
        I[ij] -= self.lr * ij_grad

    def fit(self, R, hidden_dim: int = 10, n_iters: int = 10, batch_size: int = 32, verbose: bool = True):
        n_users, n_items = R.shape
        self._initialize_matrix(n_users, n_items, hidden_dim)

        indptr = R.indptr
        indices = R.indices
        n_users, n_items = R.shape

        for step in range(n_iters):
            users, items_pos, items_neg = self._sample(n_users, n_items, batch_size, indices, indptr)
            iteration_time = self._grad_step(users, items_pos, items_neg)

            if verbose:
                mse, reg = self._calc_loss(R)
                print(f"Iteration: {step + 1}; time: {iteration_time:.1f}; loss: {mse + reg:.2f}; mse: {mse:.2f}")

    def _sample(self, n_users, n_items, batch_size, indices, indptr):
        """sample batches of random triplets u, i, j"""
        sampled_pos_items = np.zeros(batch_size, dtype = np.int)
        sampled_neg_items = np.zeros(batch_size, dtype = np.int)
        sampled_users = np.random.choice(
            n_users, size=batch_size, replace=False)

        for idx, user in enumerate(sampled_users):
            pos_items = indices[indptr[user]:indptr[user + 1]]
            pos_item = np.random.choice(pos_items)
            neg_item = np.random.choice(n_items)
            while neg_item in pos_items:
                neg_item = np.random.choice(n_items)

            sampled_pos_items[idx] = pos_item
            sampled_neg_items[idx] = neg_item

        return sampled_users, sampled_pos_items, sampled_neg_items
