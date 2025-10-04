import random
import math
import time
import matplotlib.pyplot as plt
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
        cost += graph[tour[i], tour[i + 1]]
    cost += graph[tour[-1], tour[0]]
    return cost
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
def randomTour(graph: dict, n = None) -> list:
    if n is None:
        if not graph:
            raise ValueError("Graph is empty, cannot infer number of nodes.")
        nodes = set()
        for (i, j) in graph.keys():
            nodes.add(i)
            nodes.add(j)
        n = max(nodes) + 1
    tour = list(range(len(graph)))
    random.shuffle(tour)
    return tour
def choiceByEntropy(graph: dict, curr: int, unvisited: set, beta = 10.0, n = None) -> list:
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
        candidates.append((h, graph[curr, c], c))
    return candidates[0][2]
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
        nxt = choiceByEntropy(graph, curr, unvisited, beta)
        tour.append(nxt)
        visited.add(nxt)
        curr = nxt
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
    if len(visited) == len(nodes):
        return cost + graph.get((curr, start), math.inf), [curr, start]
    minCost = math.inf
    bestPath = []
    for node in nodes:
        if node not in visited:
            costs = graph[(curr, node)]
            newVisit = visited | {node}
            totalCost, path = recursiveSolver(graph, start, newVisit, node, cost + costs)
            if totalCost < minCost:
                minCost = totalCost
                bestPath = [curr] + path
    return minCost, bestPath
MAP = {}
def graph(vertices, graph=MAP) -> dict:
    for i in range(vertices):
        for j in range(i + 1, vertices):
            cost = random.randint(1, vertices)
            graph[(i, j)] = cost
            graph[(j, i)] = cost
    return graph
inputSize = list(range(5, 20))
entropyTime = []
twoOptTime = []
NNHueristicTime = []
RecursionTime = []
for i in range(5, 20):
    graph(i)
    startNode = random.randint(0, i - 1)
    stTime = time.perf_counter()
    tourList = solver(MAP, startNode)
    cost = tour(tourList, MAP)
    enTime = time.perf_counter()
    entropyTime.append(enTime - stTime)
    stTime = time.perf_counter()
    newTour = twoOpt(tourList, MAP)
    minCost = math.inf
    while newTour[0] < minCost:
        minCost = newTour[0]
        newTour = twoOpt(newTour[1], MAP)
    enTime = time.perf_counter()
    twoOptTime.append(enTime - stTime)
    stTime = time.perf_counter()
    nnSolver = nn(MAP, startNode)
    nnSolverCost = tour(nnSolver, MAP)
    enTime = time.perf_counter()
    NNHueristicTime.append(enTime - stTime)
    stTime = time.perf_counter()
    recursiveTour = recursiveSolver(MAP, startNode)
    enTime = time.perf_counter()
    RecursionTime.append(enTime - stTime)
plt.figure(figsize=(10, 6))
plt.plot(inputSize, entropyTime, label='Entropy Solver', color = 'b')
plt.plot(inputSize, twoOptTime, label='Recursive 2-Opt', color = 'g')
plt.plot(inputSize, NNHueristicTime, label='Nearest Neighbor', color = 'r')
plt.plot(inputSize, RecursionTime, label='Recursive Solver', color = 'k')
plt.xlabel("Input Size (n)")
plt.ylabel("Execution Time (miliseconds)")
plt.yscale('log')
plt.title("TSP Algorithms: Time vs Input Size")
plt.legend()
plt.grid(True)
plt.show()