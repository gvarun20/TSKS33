import math
import numpy as np
import scipy.sparse.linalg
from matplotlib import pyplot as plt

from load_data import load_data, getA, nr_users, nr_movies


def clamp(n, minimum, maximum):
    return max(min(n, maximum), minimum)


def cal_RMSE(r_hat, r):
    return math.sqrt(np.mean((r - r_hat) ** 2))


def draw_histogram(r_hat, r, r_hat_to_compare=None, title=""):
    err = abs(np.round(r_hat) - r)
    plt.figure()

    if r_hat_to_compare is not None:
        compare_err = abs(np.round(r_hat_to_compare) - r)
        plt.hist(compare_err, edgecolor="blue", bins=np.arange(-0.5, 5), alpha=0.5, label="Comparison")

    plt.hist(err, edgecolor="black", bins=np.arange(-0.5, 5), alpha=0.7, label="Prediction")
    plt.title(title)
    plt.legend()


class BaseLinePredictor:
    def train(self, data):
        A = getA(data)
        M = len(data)

        self.r_bar = data[:, 2].sum() / M
        C = data[:, 2] - self.r_bar
        B = scipy.sparse.linalg.lsqr(A, C)[0]

        self.bu = B[:nr_users]
        self.bm = B[nr_users:]

    def predict(self, data):
        r_hat = np.zeros(len(data))
        for i, (user, movie, _) in enumerate(data):
            r_hat[i] = clamp(self.r_bar + self.bu[user] + self.bm[movie], 1, 5)
        return r_hat


class NeighborhoodPredictor(BaseLinePredictor):
    def __init__(self, u_min, L, filename=""):
        super().__init__()
        self._u_min = u_min
        self._L = L
        self.filename = filename

    def train(self, data):
        super().train(data)

        self.r_tilde = np.zeros((len(self.bu), len(self.bm)))
        for user, movie, rating in data:
            self.r_tilde[user][movie] = rating - clamp(self.r_bar + self.bu[user] + self.bm[movie], 1, 5)

        # Compute the similarity matrix D
        self.D = self.delta_matrix()

        # Validation code for D matrix
        if self.filename == "verification" and self._u_min == 20:
            verification_D = np.load("verification_D_mat.npy")
            error_in_D = np.linalg.norm(verification_D - self.D)
            print(f"Error in D matrix: {error_in_D:.5f}")
            if error_in_D < 1e-5:
                print("D matrix verification passed.")
            else:
                print("D matrix verification failed.")

    def predict(self, data):
        r_hat = np.zeros(len(data))
        delta_matrix = self.D

        last_movie = -1
        correction_denominator = 0

        for i, (user, movie, _) in enumerate(data):
            to_be_summed = [
                (delta_matrix[movie][m] * self.r_tilde[user][m], abs(delta_matrix[movie][m]))
                for m in range(nr_movies)
                if m != movie
            ]
            to_be_summed.sort(key=lambda x: x[1], reverse=True)

            correction_numerator = 0
            correction_term = 0

            if movie != last_movie:
                correction_denominator = 0
                for k, (temp_num, temp_den) in enumerate(to_be_summed):
                    if k > self._L or temp_den == 0:
                        break
                    correction_numerator += temp_num
                    correction_denominator += temp_den
            else:
                for k, (temp_num, _) in enumerate(to_be_summed):
                    if k > self._L:
                        break
                    correction_numerator += temp_num

            if correction_denominator != 0:
                correction_term = correction_numerator / correction_denominator

            last_movie = movie
            r_hat[i] = clamp(self.r_bar + self.bu[user] + self.bm[movie] + correction_term, 1, 5)

        return r_hat

    def delta(self, i, j, common_users):
        u = np.nonzero(common_users)[0]
        if len(u) < self._u_min:
            return 0

        num = np.dot(self.r_tilde[:, i], self.r_tilde[:, j])
        den = (np.linalg.norm(self.r_tilde[:, i][u]) * np.linalg.norm(self.r_tilde[:, j][u]))
        return num / den if den != 0 else 0

    def delta_matrix(self):
        delta = np.zeros((nr_movies, nr_movies))
        A = np.copy(self.r_tilde)

        for i in range(nr_movies):
            for j in range(i + 1, nr_movies):
                if i == j:
                    delta[i][j] = 1
                    continue

                common_users = A[:, i] * A[:, j]
                delta[i][j] = self.delta(i, j, common_users)
                delta[j][i] = delta[i][j]

        return delta


if __name__ == "__main__":
    filename = "verification"
    training_data = load_data(f"{filename}.training")
    test_data = load_data(f"{filename}.test")

    print("-- Baseline Predictor --")
    baseline_predictor = BaseLinePredictor()
    baseline_predictor.train(training_data)

    r_hat_baseline_training = baseline_predictor.predict(training_data)
    r_hat_baseline_test = baseline_predictor.predict(test_data)

    rmse_baseline_training = cal_RMSE(r_hat_baseline_training, training_data[:, 2])
    rmse_baseline_test = cal_RMSE(r_hat_baseline_test, test_data[:, 2])
    print(f"Training RMSE: {rmse_baseline_training:.3f}")
    print(f"Test RMSE: {rmse_baseline_test:.3f}")

    draw_histogram(r_hat_baseline_training, training_data[:, 2], title="Baseline Training")
    draw_histogram(r_hat_baseline_test, test_data[:, 2], title="Baseline Test")
    plt.show()

    u_min, L = 20, 100
    print(f"\n-- Movie Neighborhood Predictor (u_min = {u_min}, L = {L}) --")
    neighborhood_predictor = NeighborhoodPredictor(u_min, L, filename)
    neighborhood_predictor.train(training_data)

    r_hat_neighborhood_training = neighborhood_predictor.predict(training_data)
    r_hat_neighborhood_test = neighborhood_predictor.predict(test_data)

    rmse_neighborhood_training = cal_RMSE(r_hat_neighborhood_training, training_data[:, 2])
    rmse_neighborhood_test = cal_RMSE(r_hat_neighborhood_test, test_data[:, 2])

    print(f"Training RMSE: {rmse_neighborhood_training:.3f}")
    print(f"Test RMSE: {rmse_neighborhood_test:.3f}")
    print(f"Training improvement: {(rmse_baseline_training - rmse_neighborhood_training) / rmse_baseline_training * 100:.3f}%")
    print(f"Test improvement: {(rmse_baseline_test - rmse_neighborhood_test) / rmse_baseline_test * 100:.3f}%")

    draw_histogram(r_hat_neighborhood_training, training_data[:, 2], r_hat_baseline_training, title="Comparison Training")
    draw_histogram(r_hat_neighborhood_test, test_data[:, 2], r_hat_baseline_test, title="Comparison Test")
    plt.show()