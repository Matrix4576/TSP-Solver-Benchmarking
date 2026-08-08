from typing import List
from tsp_solvers import *

n = int(input("Enter the number of vertices: "))
st = int(input("Enter the start node (0 -> num): "))
MAP = generate_graph(n)
visualize(MAP)
def probDist(graph: dict) -> dict:
    nodeDistPair = {
        u: [0] * (n - 1)
        for u in range(n)
    }
    for (u, v), dist in graph.items():
        index = v if v < u else v - 1
        nodeDistPair[u][index] = dist
    probDict = {}
    for node, distList in nodeDistPair.items():
        if node not in probDict.keys():
            probDict[node] = probability(distList, beta=0.75)
    return probDict
# Baseline Solvers:
nearestNeighbourTour = nn(MAP, st)
print(f"Cost using NN : {tour(nearestNeighbourTour, MAP)}")
twoOptCost = twoOpt(nearestNeighbourTour, MAP) # Nearest Neighbour + 2-Opt
minCost = float('inf')
while twoOptCost[0] < minCost: # type: ignore
    minCost = twoOptCost[0] # type: ignore
    twoOptCost = twoOpt(twoOptCost[1], MAP) # type: ignore
print(f"Cost using NN + 2-Opt : {twoOptCost[0]}") # type: ignore
RandomTourPlusTwoOpt = twoOpt(randomTour(MAP, st), MAP)
minCost = float('inf')
while RandomTourPlusTwoOpt[0] < minCost: # type: ignore
    minCost = RandomTourPlusTwoOpt[0] # type: ignore
    RandomTourPlusTwoOpt = twoOpt(RandomTourPlusTwoOpt[1], MAP) # type: ignore
print(f"Cost using random tour + 2-Opt : {RandomTourPlusTwoOpt[0]}") # type: ignore
entropyCost = tour(solver(MAP, st), MAP)
print(f"Cost using Entropy : {entropyCost}")