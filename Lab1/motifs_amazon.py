#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 Hands-on session 1

Erik G. Larsson 2018-2020
David Nordlund, Martin Andersson, edited 2024
"""

import sys
import snap

TPATH = "/courses/TSKS33/ht2024/data/"

G = snap.LoadEdgeList(snap.PUNGraph, TPATH + "amazon0302.txt", 0, 1)

t = snap.GetTriadsAll(G,-1)
print ("Number of triangles", t[0])
print ("Number of connected triples",t[0]*3+t[2])
print ("Number of three-stars", 143768700)

import numpy as np
import scipy.sparse as sp

Edgelist = np.genfromtxt(TPATH + 'amazon0302.txt', dtype=int)
if np.min(Edgelist) == 1:
    Edgelist -= 1 # python is 0 indexed
N = np.max(Edgelist)+1


# undirected
#"""
A_rows = np.concatenate((Edgelist[:,1], Edgelist[:,0])) 
A_cols = np.concatenate((Edgelist[:,0], Edgelist[:,1])) 
#"""
# directed
"""
A_rows = Edgelist[:,1]
A_cols = Edgelist[:,0]
"""

A_rows,A_cols = zip(*set(zip(A_rows,A_cols))) # remove duplicates
d = np.ones(len(A_rows))

A = sp.csr_array((d,(A_rows,A_cols)),shape=(N,N))
