import random
import math
import matplotlib.pyplot as plt
import networkx as nx
def probability(distances: list, beta: float) -> list:
    if not distances:
        return []
    probabilityList = []
    for d in distances:
        probabilityList.append(-beta * d)
    mx = max(probabilityList)
    exp = []
    for prob in probabilityList:
        exp.append(math.exp(prob - mx))
    s = sum(exp)
    returnLst = []
    for e in exp:
        returnLst.append(e / s)
    return returnLst
def entropy(probList: list) -> float:
    h = 0.0
    for p in probList:
        if p > 0:
            h -= p * math.log2(p)
    return h
def tour(tour: list, graph: dict):
    if len(tour) <= 1:
        return 0.0
    cost = 0.0
    for i in range(len(tour) - 1):
        cost += graph[(tour[i], tour[i + 1])]
    cost += graph[(tour[-1], tour[0])]
    return cost
def choiceByEntropy(graph: dict, curr: int, unvisited: set, beta = 0.25) -> list:
    candidates = []
    for c in unvisited:
        others = []
        for v in unvisited:
            if v != c:
                others.append(v)
        if not others:
            h = 0.0
        else:
            dists = []
            for v in others:
                dists.append(graph[curr, v])
            prob = probability(dists, beta)
            h = entropy(prob)
        candidates.append((h, graph[(curr, c)], c)) # type: ignore
    return sorted(candidates)[0][2]
def solver(graph: dict, start: int = 0, beta = 10.0, n = None):
    if n is None:
        if not graph:
            raise ValueError("Graph is empty, cannot infer number of nodes.")
        nodes = set()
        for (i, j) in graph.keys():
            nodes.add(i)
            nodes.add(j)
        n = max(nodes) + 1
    nodes = list(range(n))
    visited = {start}
    tour = [start]
    curr = start
    while len(visited) < n:
        unvisited = []
        for v in nodes:
            if v not in visited:
                unvisited.append(v)
        nxt = choiceByEntropy(graph, curr, unvisited, beta) # type: ignore
        tour.append(nxt) # type: ignore
        visited.add(nxt) # type: ignore
        curr = nxt
    return tour
def nn(graph: dict, start: int, n = None):
    if n is None:
        if not graph:
            raise ValueError("Graph is empty, cannot infer number of nodes.")
        nodes = set()
        for (i, j) in graph.keys():
            nodes.add(i)
            nodes.add(j)
        n = max(nodes) + 1
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    curr = start
    while unvisited:
        nxt = min(unvisited, key = lambda v: graph[curr, v])
        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    return tour
def randomTour(graph: dict, start: int, n=None) -> list:
    if n is None:
        nodes = {i for edge in graph for i in edge}
        n = max(nodes) + 1
    tour = list(range(n))
    tour.remove(start)
    random.shuffle(tour)
    tour = [start] + tour
    return tour
def twoOpt(tours, graph, n = None):
    if n is None:
        if not graph:
            raise ValueError("Graph is empty, cannot infer number of nodes.")
        nodes = set()
        for (i, j) in graph.keys():
            nodes.add(i)
            nodes.add(j)
        n = max(nodes) + 1
    improved = True
    best = tours
    bestCost = tour(tours, graph)
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                newTour = best[:i] + best[i:j][::-1] + best[j:]
                newCost = tour(newTour, graph)
                if newCost < bestCost:
                    best = newTour
                    bestCost = newCost
                    improved = True
                    break
            if improved:
                break
        return bestCost, best
def recursiveSolver(graph: dict, start: int, visited = None, curr = None, cost = 0, n = None):
    if n is None:
        if not graph:
            raise ValueError("Graph is empty, cannot infer number of nodes.")
        nodes = set()
        for (i, j) in graph.keys():
            nodes.add(i)
            nodes.add(j)
        n = max(nodes) + 1
    if visited is None:
        visited = set([start])
        curr = start
    if len(visited) == len(nodes): # type: ignore
        return cost + graph.get((curr, start), math.inf), [curr, start]
    minCost = math.inf
    bestPath = []
    for node in nodes: # type: ignore
        if node not in visited:
            costs = graph[(curr, node)]
            newVisit = visited | {node}
            totalCost, path = recursiveSolver(graph, start, newVisit, node, cost + costs)
            if totalCost < minCost:
                minCost = totalCost
                bestPath = [curr] + path
    return minCost, bestPath
MAP = {}
def generate_graph(vertices, graph=MAP) -> dict:
    for i in range(vertices):
        for j in range(i + 1, vertices):
            cost = random.randint(1, 50)
            graph[(i, j)] = cost
            graph[(j, i)] = cost
    return graph
def visualize(graph: dict):
    G = nx.Graph()
    for (u, v), dist in graph.items():
        if u < v:
            G.add_edge(u, v, weight=dist)
    pos = nx.kamada_kawai_layout(G, weight="weight")
    plt.figure(figsize=(10, 8))
    plt.title("Graph Representation")
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=600)
    nx.draw_networkx_labels(
        G, pos, font_size=12, font_family="sans-serif", font_weight="bold"
    )
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
# n = int(input('Enter the number of vertices: '))
# startNode = int(input('Enter the starting point: '))
# MAP = generate_graph(n)
# visualize(MAP)
# entropySolver = solver(MAP, startNode)
# entropySolverCost = tour(entropySolver, MAP)
# print('Using Entropy: ', entropySolver)
# print('Using Entropy, cost: ', entropySolverCost)
# nnSolver = nn(MAP, startNode)
# nnSolverCost = tour(nnSolver, MAP)
# print('Using NN Hueristic: ', nnSolver)
# print('Using NN Hueristic, cost: ', nnSolverCost)
# twoOptSolver = twoOpt(randomTour(MAP, startNode), MAP)
# minCost = math.inf
# while twoOptSolver[0] < minCost: # type: ignore
#     minCost = twoOptSolver[0] # type: ignore
#     twoOptSolver = twoOpt(twoOptSolver[1], MAP) # type: ignore
# print('Using 2 Opt Improvement: ', twoOptSolver[1]) # type: ignore
# print('Using 2 Opt Improvement, cost: ', twoOptSolver[0]) # type: ignore
# recursionSolver = recursiveSolver(MAP, startNode)
# print('Using Recursion: ', recursionSolver[1])
# print('Using Recursion, cost: ', recursionSolver[0])