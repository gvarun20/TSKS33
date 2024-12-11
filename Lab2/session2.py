import math
import numpy as np
from scipy.sparse.linalg import lsqr
from matplotlib import pyplot as plt
from load_data import load_data, getA, nr_users, nr_movies


nr_users = 2000
nr_movies = 1500


def train_baseline(training_data):
    """
    Uses the provided dataset to train the baseline predictor.
    Returns:
    - r_bar: average rating over all users and movies.
    - bu: vector where the i:th entry represents the bias of the i:th user compared to r_bar.
    - bm: vector where the i:th entry represents the bias of the i:th movie compared to r_bar.
    """
    A = getA(training_data)
    M = len(training_data)

    r_bar = np.mean(training_data[:, 2])
    C = training_data[:, 2] - r_bar
    B = lsqr(A, C)[0]

    bu = B[:nr_users]
    bm = B[nr_users:]

    return r_bar, bu, bm


def baseline_prediction(training_data, datasets_to_predict):
    """
    Uses the training_data to train the baseline predictor,
    then evaluates its performance on all the datasets in the test_datas list.
    """
    r_bar, bu, bm = train_baseline(training_data)

    r_hats = []
    for data in datasets_to_predict:
        r_hat = np.array([
            clamp(r_bar + bu[int(user)] + bm[int(movie)], 1, 5)
            for user, movie, _ in data
        ])
        r_hats.append(r_hat)

    return r_hats


def neighborhood_prediction(training_data, datasets_to_predict, u_min=1, L=nr_movies):
    """
    Uses the training_data to train the improved predictor,
    then evaluates its performance on all the datasets in the test_datas list.
    """

    # Train the baseline model
    r_bar, bu, bm = train_baseline(training_data)

    # Compute residuals matrix r_tilde
    r_tilde = np.zeros((nr_users, nr_movies))
    for user, movie, rating in training_data:
        user, movie = int(user), int(movie)
        r_tilde[user, movie] = rating - clamp(r_bar + bu[user] + bm[movie], 1, 5)

    # Create cosine similarity matrix D
    D = np.zeros((nr_movies, nr_movies))
    for i in range(nr_movies):
        for j in range(i + 1, nr_movies):
            common_users = np.nonzero(r_tilde[:, i] * r_tilde[:, j])[0]
            if len(common_users) >= u_min:
                numerator = np.dot(r_tilde[common_users, i], r_tilde[common_users, j])
                denominator = (
                    np.linalg.norm(r_tilde[common_users, i]) *
                    np.linalg.norm(r_tilde[common_users, j])
                )
                D[i, j] = numerator / denominator if denominator != 0 else 0
                D[j, i] = D[i, j]
        D[i,i] =1       

    print(D[:5, :5])

    # Verify D matrix for the verification dataset
    if filename == "verification" and u_min == 20:
        error_in_D = np.linalg.norm(np.load('verification_D_mat.npy') - D)
        print("Error in D matrix: {0:.5f}\n".format(error_in_D))

    # Evaluate the performance of the improved predictor
    r_hats = []
    for data in datasets_to_predict:
        r_hat = np.zeros(len(data))
        for i, (user, movie, _) in enumerate(data):
            user, movie = int(user), int(movie)
            neighbors = [
                (D[movie, m] * r_tilde[user, m], abs(D[movie, m]))
                for m in range(nr_movies) if m != movie and abs(D[movie , m]) > 0
            ]
            neighbors.sort(key=lambda x: x[1], reverse=True)

            numerator = sum(n[0] for n in neighbors[:L])
            denominator = sum(n[1] for n in neighbors[:L])
            correction = numerator/ denominator if denominator !=0 else 0
                
            r_hat[i] = clamp(r_bar + bu[user] + bm[movie] + correction, 1, 5)
        r_hats.append(r_hat)

    return r_hats


def RMSE(r_hat, r):
    # Compute the RMSE between the true ratings r and the predicted ratings r_hat
    return math.sqrt(np.mean((r - r_hat) ** 2))


def draw_histogram(r_hat, r, name=""):
    # Create the described histogram
    err = abs(np.round(r_hat) - r)
    plt.hist(err, bins=np.arange(-0.5, 5), edgecolor="black", alpha=0.7)
    plt.title(name)
    plt.show()


def clamp(n, minimum, maximum):
    """
    Clamps a number between a minimum and maximum value.
    """
    return max(min(n, maximum), minimum)


filename = "task---2"  # Change this to "LiU-ID", "task---2" as needed

training_data = load_data(filename + '.training')
test_data = load_data(filename + '.test')


# ====================================== TASK 2 ======================================
print("---- baseline predictor ----")

[r_hat_baseline_training, r_hat_baseline_test] = \
    baseline_prediction(training_data, [training_data, test_data])

rmse_baseline_training = RMSE(r_hat_baseline_training, training_data[:, 2])
rmse_baseline_test = RMSE(r_hat_baseline_test, test_data[:, 2])

print("Training RMSE: {0:.3f}".format(rmse_baseline_training))
print("Test RMSE: {0:.3f}".format(rmse_baseline_test))

draw_histogram(r_hat_baseline_test, test_data[:, 2], "Baseline Test")


# ====================================== TASK 2 ======================================
u_min = 50
L = 100
print("\n---- movie neighborhood predictor with u_min = {} and L = {} ----".format(u_min, L))

[r_hat_neighborhood_training, r_hat_neighborhood_test] = \
    neighborhood_prediction(training_data, [training_data, test_data], u_min, L)

rmse_neighborhood_training = RMSE(r_hat_neighborhood_training, training_data[:, 2])
rmse_neighborhood_test = RMSE(r_hat_neighborhood_test, test_data[:, 2])

print("Training RMSE: {0:.3f}".format(rmse_neighborhood_training))
print("Test RMSE: {0:.3f}".format(rmse_neighborhood_test))

print("\nTraining Improvement: {0:.3f}%".format(
    (rmse_baseline_training - rmse_neighborhood_training) / rmse_baseline_training * 100))
print("Test Improvement: {0:.3f}%".format(
    (rmse_baseline_test - rmse_neighborhood_test) / rmse_baseline_test * 100))
