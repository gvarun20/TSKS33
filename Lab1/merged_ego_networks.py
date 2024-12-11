import snap
import sys 

sys.path.append("/courses/TSKS33/ht2024/common-functions")

from save_Gephi_NET import saveGephi_net

TPATH = "/courses/TSKS33/ht2024/data/"
G,Map = snap.LoadEdgeListStr(snap.TUNGraph, TPATH + 'DBLP.txt',Mapping=True)

print("Full DBLP network")
G.PrintInfo()

# ---- add two or three names as seeds ----
# "Emre_Telatar"
# "H._Vincent_Poor"
# "Yonina_C._Eldar"
# "Andrea_J._Goldsmith"
# "Ove_Edfors"
# "Emil_Bjornson"
# "Thomas_L._Marzetta"
# "Yann_LeCun"
names = []
# For example: names = ["Thomas_L._Marzetta","Emre_Telatar"]

selected_nodes = snap.TIntV()
for name in names:
    n = Map.GetKeyId(name)
    _,nds = G.GetNodesAtHop(n,1,False)
    nds.append(n)
    selected_nodes.AddVMerged(nds)
G_ego = snap.GetSubGraph(G,selected_nodes)

print("\nMerged ego networks of",names)
G_ego.PrintInfo()

saveGephi_net(G_ego,"DBLP.NET",Map)
