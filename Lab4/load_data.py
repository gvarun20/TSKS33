#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 Hands-on session 4, load data

vargu125
"""

import snap
from random import randint, random
from gen_data import genmod10star
from gen_data import genLiveJournal

print("test")

# -- load 10-star --
#G, h = genmod10star()

# -- load LiveJournal --
G, h = genLiveJournal()

S=10000
N= G.GetNodes()

print("Done  generation properly continue")
# Task 1------------------------------------------------------------------------------------------------------------
print ("estimates")
sum = 0
for n in G.Nodes():
    t = h[n.GetId()]
    sum+=t
exact = sum/G.GetNodes()
print("-- expected values of <x>-hat -----")
print("uniform sampling:", exact)

sum = 0
for i in G.Nodes():
    node = G.GetNI(i.GetId())
    tmp = 0
    for nbr_id in node.GetOutEdges():
        nbr = G.GetNI(nbr_id)
        t = h[nbr.GetId()]
        tmp+=t
    sum += (tmp / node.GetDeg())

print("random connection of random node:", round(sum/G.GetNodes(), 3))

sum = 0

for node in G.Nodes():
    for nbr_id in node.GetOutEdges():
        nbr = G.GetNI(nbr_id)
        sum += h[nbr.GetId()]
    
print("uniform random walk:", sum/(2*G.GetEdges()))

sum = 0
for n in G.Nodes():
    t = h[n.GetId()]
    sum+=t
print("Metro-H random walk:", sum/G.GetNodes())
#-------------------------------------------------------------------------------------------
sum = 0
for n in G.Nodes():
    sum += h[n.GetId()]
avg_x = sum / N

print("Task 1  use this valve this is correct:", avg_x)
#----------------------------------------------------------------------

#-------------------------------------------------------------

#------------------------------------------------------------


print("----------------------------------------------------------------------------------------------------------")

# Task 2
for i in range(5):
    Rnd = snap.TRnd(i*1000)
    Rnd.Randomize()
    sum = 0
    for j in range(S):
        sum += h[G.GetRndNId(Rnd)]
    result = sum / S
    print("Uniform Sampli Task 2 iter", i, ":", result)

print("------------------------------------------------------------------------------------------------------------------")

# Task 3
for i in range(5):
    sum = 0
    Rnd = snap.TRnd(randint(0, 100000000))
    Rnd.Randomize()
    for j in range(S):
        random_node = G.GetRndNId(Rnd)
        iterator = G.GetNI(random_node)
        node_degree = iterator.GetDeg()
        random_neighbor = iterator.GetNbrNId(randint(0, node_degree - 1))
        sum += h[random_neighbor]
    result = sum / S
    print("RandomConnectionONRandomNodeTask3 iter", i, ":", result)
    
print("---------------------------------------------------------------------------------------------------------------------")   

# Task 4
for i in range(5):
    sum = 0
    Rnd = snap.TRnd(randint(0, 100000000))
    Rnd.Randomize()
    random_node = G.GetRndNId(Rnd)
    iterator = G.GetNI(random_node) # Random start

    for j in range(S): # Get to a steady state
        node_degree = iterator.GetDeg()
        random_neighbor = iterator.GetNbrNId(randint(0, node_degree - 1))
        iterator = G.GetNI(random_neighbor)

    for k in range(S): # Begin sampling
        node_degree = iterator.GetDeg()
        random_neighbor = iterator.GetNbrNId(randint(0, node_degree - 1))
        iterator = G.GetNI(random_neighbor)
        sum += h[iterator.GetId()]
    result = sum / S
    print("Uniform Random Walk Task 4 iter", i, ":", result)

print("-----------------------------------------------------------------------------------------------------------------")




# Task 5

def walk(iterator):
    n_prime = iterator.GetDeg()

    random_neighbor = iterator.GetNbrNId(randint(0, n_prime - 1))

    random_neighbor_iterator = G.GetNI(random_neighbor)
    n = random_neighbor_iterator.GetDeg()

    odds = n_prime / n
    if random() < odds:
        return random_neighbor_iterator

    return iterator

for i in range(5):
    sum = 0
    Rnd = snap.TRnd(randint(0, 100000000))
    Rnd.Randomize()
    
    random_node = G.GetRndNId(Rnd)
    iterator = G.GetNI(random_node) # Random start

    for j in range(S): # Get to a steady state
        iterator = walk(iterator)

    for k in range(S): # Begin sampling
        iterator = walk(iterator)
        sum += h[iterator.GetId()]
    result = sum / S
    print("M-H random walk Task 5 iter", i, ":", result)

print("-----------------------------------------------------------------------------------")





