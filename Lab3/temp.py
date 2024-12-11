# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 

titles= open('titles/_do_not_delete_this_file.txt','r').read().strip().splitlines()
links = np.genfromtxt('links/_do_not_delete_this_file.txt', delimiter='',dtype=int)

N = len(titles)

links -=1 

A =np.zeros((N,N))
for(i,j) in links:
    A[j,i]=1
    
u=np.ones(N)

k_in =A@u
k_out =A.T @ u

k_in = k_in/np.sum(k_in)
k_out = k_out/np.sum(k_out)

s=np.argsort(k_in)[-5:].tolist()[::-1]

top5titles = [titles[i]for i in s]

print('top in-degree \t in_degrees \t out_degrees')

L= max(map(len,[titles[i]for i in s]))

for i in s:
    print('{titles:|{L}} \t {centrality:.6f} \t {centrality1:.6f} '.format(titles=titles[i],  L=L,centrality=k_in[i],centrality2=k_out[i]))