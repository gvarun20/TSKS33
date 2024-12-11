import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
from matplotlib import pyplot as plt

# Configuration for the dataset and matrix dimensions
filename = "verification"
num_users = 2000
num_movies = 1500

def load_data(file_name):
    """
    Load user-movie ratings data from a CSV file.
    Parameters:file_name (str): Path to the CSV file.
    Returns:np.ndarray: Array of shape (num_ratings, 3) with user IDs, movie IDs, and ratings.
    """
    # Load data as integers and adjust IDs to be zero-indexed
    data = np.genfromtxt(file_name, delimiter=',', dtype=int)
    data[:, 0:2] -= 1  # Convert user and movie IDs from 1-based to 0-based indexing
    return data

def build_user_movie_matrix(ratings):
    """
    Construct a sparse user-movie interaction matrix.
    Parameters:ratings (np.ndarray): Array of shape (num_ratings, 3) with user IDs, movie IDs, and ratings.
    Returns:sparse.csr_matrix: Sparse matrix where rows represent ratings and columns represent user/movie biases.
    """
    num_ratings = len(ratings)
    # Row indices (one for each rating)
    row_indices = np.concatenate((np.arange(num_ratings, dtype=int), np.arange(num_ratings, dtype=int)))
    # Column indices (user IDs and movie IDs shifted by the number of users)
    column_indices = np.concatenate((ratings[:, 0], ratings[:, 1] + num_users))
    # Data values (all ones for creating the sparse matrix)
    data_values = np.ones((2 * num_ratings,))
    # Create a sparse matrix of shape (num_ratings, num_users + num_movies)
    user_movie_matrix = sparse.csr_matrix((data_values, (row_indices, column_indices)), shape=(num_ratings, num_users + num_movies))
    return user_movie_matrix

# Load training and test datasets
training_ratings = load_data(filename + '.training')
test_ratings = load_data(filename + '.test')

# Display the datasets for verification
print("Training Data:")
print(training_ratings)

print("\nTest Data:")
print(test_ratings)

# Build the sparse user-movie matrix for training data
user_movie_matrix = build_user_movie_matrix(training_ratings)

print("\nUser-Movie Interaction Matrix:")
print(user_movie_matrix)
