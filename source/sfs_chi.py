import numpy as np
from scipy.stats import chi
from sicore import SelectiveInferenceChi
from sicore import polytope_to_interval, intersection


class SfsSelectiveInferenceChi:
    def __init__(self, X, y, var, k, test_indexes):
        self.X = X
        self.data = y
        self.var = var
        self.k = k
        self.test_indexes = test_indexes

    def sfs(self, data):
        A = []
        A_c = list(range(self.X.shape[1]))

        for _ in range(self.k):
            X_A = self.X[:, A]
            correlation = (
                data - X_A @ np.linalg.inv(X_A.T @ X_A) @ X_A.T @ data
            ).T @ self.X[:, A_c]
            index = np.argmax(np.abs(correlation))
            A.append(A_c[index])
            A_c.remove(A_c[index])
        return A

    def construct_P(self):
        self.active_set = self.sfs(self.data)
        g_set = []
        for index in self.test_indexes:
            g_set.append(self.active_set[index])
        minus_set = list(set(self.active_set) - set(g_set))

        X_m = self.X[:, minus_set]
        diff_proj = (
            np.identity(self.X.shape[0]) - X_m @ np.linalg.inv(X_m.T @ X_m) @ X_m.T
        )
        X_g = self.X[:, g_set]

        U, _, _ = np.linalg.svd(diff_proj @ X_g, full_matrices=False)
        self.P = U @ U.T
        self.degree = int(np.trace(self.P) + 1e-3)

        assert self.degree == U.shape[1]
        # assert np.allclose(self.P, self.P.T)
        # assert np.allclose(self.P, self.P @ self.P)

        self.max_tail = chi.mean(self.degree) + 40 * np.sqrt(chi.var(self.degree))
        self.max_tail = 100

    def model_selector(self, active_set):
        return set(self.active_set) == set(active_set)

    def algorithm(self, a, b, z):
        X = self.X

        data = a + b * z
        A = self.sfs(data)
        A_c = list(range(X.shape[1]))

        intervals = [-np.inf, np.inf]

        for i in range(self.k):
            x_i = X[:, A[i]]
            X_Ai = X[:, A[0:i]]
            P_Ai = (
                np.identity(X.shape[0]) - X_Ai @ np.linalg.inv(X_Ai.T @ X_Ai) @ X_Ai.T
            )

            A_c.remove(A[i])

            x_i = np.reshape(x_i, [-1, 1])
            A_mat_i = P_Ai @ x_i @ x_i.T @ P_Ai

            for j in A_c:
                x_j = np.reshape(X[:, j], [-1, 1])
                A_mat_j = P_Ai @ x_j @ x_j.T @ P_Ai

                A_mat = A_mat_j - A_mat_i

                intervals_ij = polytope_to_interval(a, b, A=A_mat)
                intervals = intersection(intervals, intervals_ij)

        return A, intervals

    def inference(self, **kwargs):
        self.si_calculator = SelectiveInferenceChi(self.data, self.var, self.P)

        result = self.si_calculator.inference(
            self.algorithm, self.model_selector, max_tail=self.max_tail, **kwargs
        )
        return result
