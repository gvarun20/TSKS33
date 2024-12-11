#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSKS33 Hands-on session 1

Erik G. Larsson 2018-2020
"""

import sys
sys.path.append("/courses/TSKS33/ht2024/common-functions")
import snap
import os
from save_Gephi_NET import saveGephi_net

TPATH = "/courses/TSKS33/ht2024/data/"

# ==========================

# Generate a complete graph with 5 nodes (K5) and look at it via Graphviz
G = snap.GenFull(snap.PUNGraph, 5)
#for EI in G.Edges():
#    print "link from %d to %d" % (EI.GetSrcNId(), EI.GetDstNId())
    
snap.SaveGViz(G, "_undirected-completely-connected.dot", "Undirected Completely Connected Network", True)
os.system("neato -Tpdf _undirected-completely-connected.dot >_undirected-completely-connected.pdf")

# ==========================

# Generate a star graph and visualize in Graphviz
G = snap.GenStar(snap.PNGraph, 10, True)
#for EI in G.Edges():
#    print "edge: (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId())
    
snap.SaveGViz(G, "_directed-star.dot", "Directed Star Graph", True)
os.system("neato -Tpdf _directed-star.dot >_directed-star.pdf")

# ==========================

# Generate some Poisson random graphs and visualize with Gephi
G1 = snap.GenRndGnm(snap.PUNGraph, 100, 50)
G2 = snap.GenRndGnm(snap.PUNGraph, 100, 100)
G3 = snap.GenRndGnm(snap.PUNGraph, 100, 1000)

saveGephi_net(G1,"_Poisson-1.NET")
saveGephi_net(G2,"_Poisson-2.NET")
saveGephi_net(G3,"_Poisson-3.NET")

G4 = snap.GenRndGnm(snap.PUNGraph, 1000, 10000)
snap.PlotInDegDistr(G4, "_Poisson-4", "Poisson, degree distribution")

# ==========================

# Generate a scale-free network and visualize with Gephi
G1 = snap.GenPrefAttach(10, 3)
G2 = snap.GenPrefAttach(50, 5)
G3 = snap.GenPrefAttach(100, 10)

saveGephi_net(G1,"_Pref-attach-1.NET")
saveGephi_net(G2,"_Pref-attach-2.NET")
saveGephi_net(G3,"_Pref-attach-3.NET")
snap.PrintInfo(G3, "Python type PNGraph", "_Pref-attach-3.info.txt", False)

G4 = snap.GenPrefAttach(100000, 10)
snap.PlotInDegDistr(G4, "_Pref-attach-4", "Preferential attachment 4, degree distribution")

# ==========================

# Examine the 88234-edge subnetwork of the Facebook graph
# http://snap.stanford.edu/data/ego-Facebook.html
G = snap.LoadEdgeList(snap.PUNGraph, TPATH + "facebook_combined.txt", 0, 1)
snap.PrintInfo(G, "facebook", "_facebook-info.txt", False)

# ==========================

# Examine the Amazon product co-purchase network
# http://snap.stanford.edu/data/amazon0302.html
G = snap.LoadEdgeList(snap.PUNGraph, TPATH + "amazon0302.txt", 0, 1)
snap.PrintInfo(G, "amazon", "_amazon0302-info.txt", False)

# ==========================

# Examine DBLP
G = snap.LoadEdgeListStr(snap.PUNGraph, TPATH + "DBLP.txt", 0, 1,Mapping=False)
snap.PrintInfo(G, "DBLP", "_dblp-info.txt", True)

# Examine by some random sampling if the "small-world" property of DBLP seems to be true.
for i in range(1,25):
    N1 = G.GetRndNId()
    N2 = G.GetRndNId()

    L = snap.GetShortPath(G, N1, N2)
    print (L)
