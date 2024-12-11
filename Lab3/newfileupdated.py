#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:15:40 2024

@author: vargu125
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy import linalg


def getLargestIndexes(vector):
    largest_values = sorted(vector, key=abs, reverse=True)[:5]
    indexes = []
    for i in largest_values:
        number, = np.where(np.isclose(vector, i))
        indexes.append(number[0])
    return indexes



titles= open('titles/1.txt','r').read().strip().splitlines()
links = np.genfromtxt('links/1.txt', delimiter='',dtype=int) #_do_not_delete_this_file.

N = len(titles)

links -=1 

A =np.zeros((N,N))
for(i,j) in links:
    A[j,i]=1
    
u=np.ones(N)

running = (True, True, True, True, True, True)

# Task 1----------------------------------------------------------------------------------------------------
if running[0]:  
    k_in = A @ u
    k_out = A.T @ u

    k_in = k_in / np.sum(k_in)
    k_out = k_out / np.sum(k_out)

    s = np.argsort(k_in)[-5:].tolist()[::-1]

    print("Top in-degree \t in-degree \t out_degree")

    L = max(map(len, [titles[i] for i in s]))

    for i in s:
        print("{title:<{L}} \t {centrality:.6f} \t {centrality2:.6f}".format(title=titles[i], L = L, centrality=k_in[i], centrality2= k_out[i]))

    s = np.argsort(k_out)[-5:].tolist()[::-1]

    print("-------------------------------------------------------------")
    print("Top out-degree \t out-degree \t in_degree")

    L = max(map(len, [titles[i] for i in s]))

    for i in s:
        print("{title:<{L}} \t {centrality:.6f} \t {centrality2:.6f}".format(title=titles[i], L = L, centrality=k_out[i], centrality2= k_in[i]))


# Task 2-----------Calculate the hub and authority centrality of all articles. ------------------------------------------------------------------
if running[1]:    
    print("-----------------------------------------------------------")
    eigenValuesHub, eigenVectorsHub = np.linalg.eigh(A.T @ A)
    eigenValuesauth, eigenVectorsauth = np.linalg.eigh(A @ A.T)
    vector_index = len(eigenValuesHub) - 1

    dominant_hub_vector = eigenVectorsHub[:,vector_index]
    dominant_hub_vector = dominant_hub_vector / np.sum(dominant_hub_vector)
    largest_hub_indexes = getLargestIndexes(dominant_hub_vector)


    dominant_auth_vector = eigenVectorsauth[:,vector_index]
    dominant_auth_vector = dominant_auth_vector / np.sum(dominant_auth_vector)
    largest_auth_indexes = getLargestIndexes(dominant_auth_vector)


    print("Top hubs \t Hub centrality \t Authority centrality")

    L = max(map(len, [titles[i] for i in largest_hub_indexes]))

    for i in largest_hub_indexes:
        print("{title:<{L}} \t {centrality:.6f} \t {centrality2:.6f}".format(title=titles[i], L = L, centrality=dominant_hub_vector[i], centrality2= dominant_auth_vector[i]))

    print("-------------------------------------------------")

    print("Top Authorities \t Authority centrality \t Hub centrality")

    L = max(map(len, [titles[i] for i in largest_hub_indexes]))

    for i in largest_auth_indexes:
        print("{title:<{L}} \t {centrality:.6f} \t {centrality2:.6f}".format(title=titles[i], L = L, centrality=dominant_auth_vector[i], centrality2=dominant_hub_vector[i] ))


# Task 3---------Calculate the eigenvector centrality of all articles-------------------------------------------------------------------------
if running[2]:
    print("------------------------------------------------------------------")

    def eigenVector_centrality(base_vector, i, alpha):
        return np.power((alpha * A), i) @ base_vector

    def normalize(x):
        fac = abs(x).max()
        x_n = x / x.max()
        return fac, x_n

    eigenVector = np.ones((N,))

    # Calculate eigenvector and value of A iteratively
    for i in range(50):
        eigenVector = np.dot(A, eigenVector)
        eigenValue, eigenVector = normalize(eigenVector)

    eigenValues, eigenVectors = np.linalg.eigh(A)

    vector = eigenVector_centrality(eigenVector, 50, 1 / eigenValue)
    vector = vector / np.sum(vector)

    largest_indexes = getLargestIndexes(vector)

    print("Top eigenvector centrality \t Eigenvector centrality")

    L = max(map(len, [titles[i] for i in largest_indexes]))

    for i in largest_indexes:
        print("{title:<{L}} \t {centrality:.6f}".format(title=titles[i], L = L, centrality=vector[i]))


# Task 4------------------------------------------------------------------------------------------------
if running[3]:
    print("----------------------------------------------------------------------")

    def ketz_centrality_iteration(base_vector, alpha, free_factor):
        return alpha * A @ base_vector + free_factor

    vector = eigenVector
    alpha = 0.85 * 1 / abs(eigenValue)
    free_factor = np.full((N,), 1/N)
    for i in range(100):
        vector = ketz_centrality_iteration(vector, alpha, free_factor)

    vector = vector / np.sum(vector)

    largest_indexes = getLargestIndexes(vector)


    print("Top Katz \t Katz centrality")

    L = max(map(len, [titles[i] for i in largest_indexes]))

    for i in largest_indexes:
        print("{title:<{L}} \t {centrality:.6f}".format(title=titles[i], L = L, centrality=vector[i]))


# Task 5----------Calculate the Google PageRank score of each article in your network--------------------------------------------------------------
if running[4]:
    print("-------------------------------------------------------------------------------")

    k_out = A.T @ u
    H = np.zeros((N,N))
    for i in range(N): # rows
        for j in range(N): # cols
            if k_out[j] == 0:
                value = 1 / N
            else:
                value = A[i][j] / k_out[j]
            H[i][j] = value
            
    alpha = 0.85
    identity_matrix = np.identity(N)
    vector = (1 - alpha) / N * linalg.inv(identity_matrix - alpha * H) @ u
    vector = vector / np.sum(vector)

    largest_indexes = getLargestIndexes(vector)

    print("Top googlePageRank, alpha 0.85 \t PageRank centrality")

    L = max(map(len, [titles[i] for i in largest_indexes]))

    for i in largest_indexes:
        print("{title:<{L}} \t {centrality:.6f}".format(title=titles[i], L = L, centrality=vector[i]))


# Task 6
if running[5]:
    print("---------------------------------------------------------------------")
    
    # Indices of the top 3 articles from Task 5 (update as needed)
    indexes = getLargestIndexes(vector)[:5]
    print(f"Top 5 articles (indices): {indexes}")
    
    # Transition matrix for iterative computation
    G = alpha * H + ((1 - alpha) / N) * np.ones((N, N))
    vector = np.full((N,), 1 / N)  # Initialize with uniform distribution
    
    data = np.zeros((100, 5))  # Store PageRank scores for the top 3 articles over 100 iterations
    
    for i in range(100):
        vector = G @ vector  # Update PageRank vector
        vector = vector / np.sum(vector)  # Normalize
        data[i, :] = vector[indexes]  # Record the scores of the top 3 articles
    
    # Closed-form solution from Task 5 for comparison
    exact_scores = vector[indexes]
    
    # Plot the evolution of PageRank scores
    plt.figure(figsize=(10, 6))
    iterations = np.arange(1, 101)
    for j, idx in enumerate(indexes):
        plt.plot(iterations, data[:, j], label=f"Article {titles[idx]} (Iterative)", linestyle='--')
        plt.axhline(y=exact_scores[j], color=f"C{j}", linestyle='-', label=f"Article {titles[idx]} (Exact)")
    
    plt.title("PageRank Score Evolution for Top 5 Articles")
    plt.xlabel("Iteration")
    plt.ylabel("PageRank Score")
    plt.legend()
    plt.grid(True)
    plt.show()
