import numpy as np
from sicore import SelectiveInferenceNorm
from sicore import polytope_to_interval, intersection


class SfsSelectiveInferenceNorm:
    def __init__(self, X, y, var, k, test_index):
        self.X = X
        self.data = y
        self.var = var
        self.k = k
        self.test_index = test_index

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

    def construct_eta(self):
        self.active_set = self.sfs(self.data)

        X_A = self.X[:, self.active_set]
        e = np.zeros(X_A.shape[1])
        e[self.test_index] = 1
        self.eta = X_A @ np.linalg.inv(X_A.T @ X_A) @ e

        self.max_tail = np.sqrt(self.eta @ self.eta) * 20

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
        self.si_calculator = SelectiveInferenceNorm(self.data, self.var, self.eta)

        result = self.si_calculator.inference(
            self.algorithm, self.model_selector, max_tail=self.max_tail, **kwargs
        )
        return result
