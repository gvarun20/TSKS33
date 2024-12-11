import math
import numpy as np
import scipy.sparse.linalg
from matplotlib import pyplot as plt
from load import load_data, build_user_movie_matrix, num_users, num_movies

class BaselineModel:
    def train(self, ratings):
        # Get user-movie interaction matrix and total ratings
        user_movie_matrix = build_user_movie_matrix(ratings)
        total_ratings = len(ratings)

        # Compute global average rating
        self.global_avg_rating = ratings[:, 2].sum() / total_ratings

        # Calculate deviations from global average
        deviations = ratings[:, 2] - self.global_avg_rating

        # Solve least squares problem to find user and movie biases
        user_movie_bias = scipy.sparse.linalg.lsqr(user_movie_matrix, deviations)[0]
        self.user_bias = user_movie_bias[:num_users]
        self.movie_bias = user_movie_bias[num_users:]

    def predict(self, ratings):
        predictions = np.zeros(len(ratings))

        # Predict ratings using global average and biases
        for i, (user_id, movie_id, _) in enumerate(ratings):
            predictions[i] = clamp(
                self.global_avg_rating + self.user_bias[user_id] + self.movie_bias[movie_id],
                1,
                5
            )
        return predictions


class NeighborhoodModel(BaselineModel):
    def __init__(self, min_common_users, top_similar_movies):
        super().__init__()
        self.min_common_users = min_common_users
        self.top_similar_movies = top_similar_movies

    def train(self, ratings):
        super().train(ratings)
        # Initialize residuals matrix
        self.residual_matrix = np.zeros((len(self.user_bias), len(self.movie_bias)))
        # Compute residuals for each rating
        for user_id, movie_id, rating in ratings:
            baseline_prediction = clamp(
                self.global_avg_rating + self.user_bias[user_id] + self.movie_bias[movie_id],
                1,
                5
            )
            self.residual_matrix[user_id][movie_id] = rating - baseline_prediction

    def predict(self, ratings):
        predictions = np.zeros(len(ratings))
        similarity_matrix = self.compute_similarity_matrix()
        last_movie_id = -1
        correction_denominator = 0
        # Predict ratings using baseline and neighborhood correction
        for i, (user_id, movie_id, _) in enumerate(ratings):
            similarity_scores = [
                (similarity_matrix[movie_id][other_movie_id] * self.residual_matrix[user_id][other_movie_id],
                 abs(similarity_matrix[movie_id][other_movie_id]))
                for other_movie_id in range(num_movies)
                if other_movie_id != movie_id
            ]

            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            correction_numerator = 0
            correction_term = 0

            if movie_id != last_movie_id:
                correction_denominator = 0
                for k, (weighted_score, similarity) in enumerate(similarity_scores):
                    if k >= self.top_similar_movies or similarity == 0:
                        break
                    correction_numerator += weighted_score
                    correction_denominator += similarity
            else:
                for k, (weighted_score, _) in enumerate(similarity_scores):
                    if k >= self.top_similar_movies:
                        break
                    correction_numerator += weighted_score

            if correction_denominator != 0:
                correction_term = correction_numerator / correction_denominator

            last_movie_id = movie_id
            predictions[i] = clamp(
                self.global_avg_rating + self.user_bias[user_id] + self.movie_bias[movie_id] + correction_term,
                1,
                5
            )
        return predictions

    def compute_similarity(self, movie_1, movie_2, common_users_mask):
        common_users = np.nonzero(common_users_mask)[0]
        if len(common_users) < self.min_common_users:
            return 0

        numerator = np.dot(self.residual_matrix[:, movie_1], self.residual_matrix[:, movie_2])
        denominator = (
            np.linalg.norm(self.residual_matrix[:, movie_1][common_users]) *
            np.linalg.norm(self.residual_matrix[:, movie_2][common_users])
        )
        return numerator / denominator if denominator != 0 else 0

    def compute_similarity_matrix(self):
        similarity_matrix = np.zeros((num_movies, num_movies))

        for movie_1 in range(num_movies):
            for movie_2 in range(movie_1 + 1, num_movies):
                if movie_1 == movie_2:
                    similarity_matrix[movie_1][movie_2] = 1
                    continue

                common_users_mask = self.residual_matrix[:, movie_1] * self.residual_matrix[:, movie_2]
                similarity_matrix[movie_1][movie_2] = self.compute_similarity(movie_1, movie_2, common_users_mask)
                similarity_matrix[movie_2][movie_1] = similarity_matrix[movie_1][movie_2]

        return similarity_matrix


def calculate_rmse(predictions, actual_ratings):
    return math.sqrt(np.mean((actual_ratings - predictions) ** 2))

def plot_histogram(predictions, actual_ratings, comparison_predictions=None, title=""):
    errors = abs(np.round(predictions) - actual_ratings)
    plt.figure()

    if comparison_predictions is not None:
        comparison_errors = abs(np.round(comparison_predictions) - actual_ratings)
        plt.hist(comparison_errors, edgecolor="blue", bins=np.arange(-0.5, 5), alpha=0.5, label="Comparison")

    plt.hist(errors, edgecolor="black", bins=np.arange(-0.5, 5), alpha=0.7, label="Prediction")
    plt.title(title)
    plt.legend()


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


if __name__ == "__main__":
    dataset_name = "task---2"
    training_ratings = load_data(f"{dataset_name}.training")
    test_ratings = load_data(f"{dataset_name}.test")

    print("-- Baseline Model --")
    baseline_model = BaselineModel()
    baseline_model.train(training_ratings)

    baseline_train_predictions = baseline_model.predict(training_ratings)
    baseline_test_predictions = baseline_model.predict(test_ratings)

    baseline_train_rmse = calculate_rmse(baseline_train_predictions, training_ratings[:, 2])
    baseline_test_rmse = calculate_rmse(baseline_test_predictions, test_ratings[:, 2])

    print(f"Training RMSE: {baseline_train_rmse:.3f}")
    print(f"Test RMSE: {baseline_test_rmse:.3f}")

    plot_histogram(baseline_train_predictions, training_ratings[:, 2], title="Baseline Training")
    plot_histogram(baseline_test_predictions, test_ratings[:, 2], title="Baseline Test")
    plt.show()

    min_common_users, top_similar_movies = 20, 100
    print(f"\n-- Neighborhood Model (min_common_users = {min_common_users}, top_similar_movies = {top_similar_movies}) --")
    neighborhood_model = NeighborhoodModel(min_common_users, top_similar_movies)
    neighborhood_model.train(training_ratings)

    neighborhood_train_predictions = neighborhood_model.predict(training_ratings)
    neighborhood_test_predictions = neighborhood_model.predict(test_ratings)

    neighborhood_train_rmse = calculate_rmse(neighborhood_train_predictions, training_ratings[:, 2])
    neighborhood_test_rmse = calculate_rmse(neighborhood_test_predictions, test_ratings[:, 2])

    print(f"Training improvement: {(baseline_train_rmse - neighborhood_train_rmse) / baseline_train_rmse * 100:.3f}%")
    print(f"Test improvement: {(baseline_test_rmse - neighborhood_test_rmse) / baseline_test_rmse * 100:.3f}%")

    plot_histogram(neighborhood_train_predictions, training_ratings[:, 2], baseline_train_predictions, title="Comparison Training")
    plot_histogram(neighborhood_test_predictions, test_ratings[:, 2], baseline_test_predictions, title="Comparison Test")
    plt.show()
