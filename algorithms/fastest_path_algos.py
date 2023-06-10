import heapq
import math
import random
from collections import defaultdict
from collections import deque

####################################
#### Dijkstra's Algorithm ####
####################################
"""
Dijkstra's Algorithm: Dijkstra's algorithm is a classic and widely-used 
algorithm for finding the shortest path between nodes in a graph. It is 
efficient and guarantees the shortest path in a graph with non-negative 
edge weights.

"""
def dijkstra(graph, start, end):
    queue = [(0, start)]
    distances = {start: 0}
    previous = {}

    while queue:
        (curr_dist, curr_node) = heapq.heappop(queue)

        if curr_node == end:
            path = []
            while curr_node is not None:
                path.append(curr_node)
                curr_node = previous.get(curr_node)
            return path[::-1]

        for neighbor, weight in graph[curr_node].items():
            new_dist = curr_dist + weight
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = curr_node
                heapq.heappush(queue, (new_dist, neighbor))

    return None


####################################
#### A* Algorithm ####
####################################
"""
A* Algorithm: A* is an informed search algorithm that combines the best 
features of Dijkstra's algorithm and the Breadth-First Search. It uses a 
heuristic function to estimate the cost to reach the goal, which allows it 
to explore the search space more efficiently.

"""
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(graph, start, end):
    queue = [(0, start)]
    g_scores = {start: 0}
    f_scores = {start: heuristic(start, end)}
    came_from = {}

    while queue:
        current = heapq.heappop(queue)[1]

        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = came_from.get(current)
            return path[::-1]

        for neighbor in graph[current]:
            tentative_g_score = g_scores[current] + 1
            if tentative_g_score < g_scores.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_scores[neighbor] = tentative_g_score
                f_scores[neighbor] = tentative_g_score + heuristic(neighbor,
                                                                   end)
                heapq.heappush(queue, (f_scores[neighbor], neighbor))

    return None


####################################
#### Bellman-Ford Algorithm ####
####################################
"""
Bellman-Ford Algorithm: The Bellman-Ford algorithm is capable of handling 
graphs with negative edge weights. It can detect negative weight cycles, 
making it a valuable tool for applications where negative weights are allowed.

"""

def bellman_ford(graph, start, end):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {}

    for _ in range(len(graph) - 1):
        for node, edges in graph.items():
            for neighbor, weight in edges.items():
                new_dist = distances[node] + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = node

    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            if distances[node] + weight < distances[neighbor]:
                raise ValueError("Negative weight cycle detected")

    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous.get(current)
    return path[::-1]


####################################
#### Floyd-Warshall Algorithm ####
####################################
"""
Floyd-Warshall Algorithm: The Floyd-Warshall algorithm is an all-pairs shortest 
path algorithm that finds the shortest paths between all pairs of vertices in 
a graph. It is especially efficient for dense graphs, where the number of 
edges is large compared to the number of vertices.

"""
def floyd_warshall(graph):
    dist = {}

    for node in graph:
        dist[node] = {}
        for neighbor in graph:
            dist[node][neighbor] = graph[node].get(neighbor, float('inf'))

    for k in graph:
        for i in graph:
            for j in graph:
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


####################################
#### Johnson's Algorithm ####
####################################
"""
Johnson's Algorithm: Johnson's algorithm combines Dijkstra's and the 
Bellman-Ford algorithms to find the shortest paths between all pairs of 
vertices in a sparse graph, even with negative edge weights, as long as 
there are no negative weight cycles.

"""

def bellman_ford_with_source(graph, source):
    distances = {node: float('inf') for node in graph}
    distances[source] = 0

    for _ in range(len(graph) - 1):
        for node, edges in graph.items():
            for neighbor, weight in edges.items():
                if distances[node] != float('inf') and distances[
                    node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight

    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            if distances[node] != float('inf') and distances[node] + weight < \
                    distances[neighbor]:
                raise ValueError("Negative weight cycle detected")

    return distances


def johnson(graph):
    source = 'source'
    temp_graph = graph.copy()
    temp_graph[source] = {node: 0 for node in graph}

    h = bellman_ford_with_source(temp_graph, source)
    del temp_graph[source]

    new_graph = defaultdict(dict)
    for u, edges in graph.items():
        for v, weight in edges.items():
            new_graph[u][v] = weight + h[u] - h[v]

    dist = {}
    for u in graph:
        dist[u] = dijkstra(new_graph, u)

    return dist


####################################
#### Bidirectional Search ####
####################################
"""
Bidirectional Search: Bidirectional search involves running two simultaneous 
searches, one forward from the start and one backward from the goal. The 
searches meet in the middle, which can significantly speed up the search 
process, especially in large graphs.

"""
def bidirectional_search(graph, start, end):
    if start == end:
        return [start]

    visited_fwd = {start}
    visited_bwd = {end}
    queue_fwd = deque([(start, [start])])
    queue_bwd = deque([(end, [end])])

    while queue_fwd and queue_bwd:
        current_fwd, path_fwd = queue_fwd.popleft()
        current_bwd, path_bwd = queue_bwd.popleft()

        if current_fwd in visited_bwd:
            return path_fwd[:-1] + path_bwd[::-1]
        if current_bwd in visited_fwd:
            return path_fwd + path_bwd[:-1][::-1]

        for neighbor in graph[current_fwd]:
            if neighbor not in visited_fwd:
                visited_fwd.add(neighbor)
                queue_fwd.append((neighbor, path_fwd + [neighbor]))

        for neighbor in graph[current_bwd]:
            if neighbor not in visited_bwd:
                visited_bwd.add(neighbor)
                queue_bwd.append((neighbor, path_bwd + [neighbor]))

    return None


####################################
#### Contraction Hierarchies ####
####################################
"""
Contraction Hierarchies: Contraction Hierarchies is a technique for 
preprocessing a graph to accelerate shortest-path queries. It involves 
contracting nodes in a hierarchical order and adding shortcut edges to 
maintain shortest-path distances between the remaining nodes.

"""
def preprocess_contraction_hierarchy(graph):
    order = {node: 0 for node in graph}
    shortcut_count = {node: 0 for node in graph}
    level = 0
    stack = []

    def edge_diff(node):
        return len(graph[node]) - shortcut_count[node]

    while order:
        min_node = min(order, key=edge_diff)
        level += 1
        order[min_node] = level
        del order[min_node]

        for pred, pred_weight in graph[min_node].items():
            for succ, succ_weight in graph[min_node].items():
                if pred != succ:
                    if graph[pred].get(succ) is None or graph[pred][
                        succ] > pred_weight + succ_weight:
                        graph[pred][succ] = pred_weight + succ_weight
                        shortcut_count[pred] += 1
                        shortcut_count[succ] += 1

        stack.append(min_node)

    return stack, level


def contraction_hierarchy_query(graph, start, end, stack, level):
    forward_queue = [(0, start)]
    backward_queue = [(0, end)]

    forward_visited = {}
    backward_visited = {}

    def process_queue(queue, visited, other_visited):
        cost, node = heapq.heappop(queue)

        if node in other_visited:
            return cost + other_visited[node]

        if node not in visited:
            visited[node] = cost
            for neighbor, weight in graph[node].items():
                if level[node] > level[neighbor]:
                    heapq.heappush(queue, (cost + weight, neighbor))

        return None

    while forward_queue and backward_queue:
        result = process_queue(forward_queue, forward_visited,
                               backward_visited)
        if result:
            return result

        result = process_queue(backward_queue, backward_visited,
                               forward_visited)
        if result:
            return result

    return None


# Sample graph
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

# Preprocessing and query
stack, level = preprocess_contraction_hierarchy(graph)
shortest_path = contraction_hierarchy_query(graph, 'A', 'D', stack, level)

print("Shortest path from 'A' to 'D':", shortest_path)


####################################
#### ALT (A*, Landmarks, and Triangle inequality) algorithm ####
####################################

"""
ALT (A*, Landmarks, and Triangle inequality): The ALT algorithm is an 
improvement on the A* algorithm. It introduces landmarks and leverages 
the triangle inequality to produce better heuristic functions, leading to 
faster search times.

"""

def heuristic(a, b, landmarks):
    return max([abs(l[a] - l[b]) for l in landmarks])


def generate_landmarks(graph, count):
    landmarks = []

    for _ in range(count):
        landmark = {node: float('inf') for node in graph}
        start = min(landmark, key=landmark.get)
        queue = [(0, start)]

        while queue:
            cost, node = heapq.heappop(queue)

            if cost < landmark[node]:
                landmark[node] = cost

                for neighbor, weight in graph[node].items():
                    heapq.heappush(queue, (cost + weight, neighbor))

        landmarks.append(landmark)

    return landmarks


def alt(graph, start, end, landmarks):
    visited = {}
    queue = [(0, start)]

    while queue:
        cost, node = heapq.heappop(queue)

        if node == end:
            return cost

        if node not in visited:
            visited[node] = cost

            for neighbor, weight in graph[node].items():
                heapq.heappush(queue, (
                    cost + weight + heuristic(node, neighbor, landmarks),
                    neighbor))

    return None


# Sample graph
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

# Generate landmarks and find shortest path
landmarks = generate_landmarks(graph, 2)
shortest_path = alt(graph, 'A', 'D', landmarks)

print("Shortest path from 'A' to 'D':", shortest_path)


####################################
#### Ant Colony Optimization (ACO) algorithm ####
####################################
"""
Ant Colony Optimization (ACO): ACO is a nature-inspired metaheuristic 
algorithm that simulates the behavior of ants searching for the shortest path 
between their nest and a food source. It is useful for finding good approximate 
solutions to complex optimization problems, such as the Traveling Salesman 
Problem.

"""
def distance(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def generate_graph(nodes):
    return [[distance(a, b) for b in nodes] for a in nodes]


def aco_tsp(graph, ants, iterations, alpha, beta, evaporation):
    num_nodes = len(graph)
    pheromones = [[1 for _ in range(num_nodes)] for _ in range(num_nodes)]

    def select_next_node(node, unvisited, pheromones, alpha, beta):
        weights = [(pheromones[node][next_node] ** alpha) * (
                (1 / graph[node][next_node]) ** beta)
                   for next_node in unvisited]
        total = sum(weights)
        probabilities = [weight / total for weight in weights]

        return random.choices(unvisited, probabilities)[0]

    def update_pheromones(pheromones, ant_tours, evaporation):
        for i in range(len(pheromones)):
            for j in range(len(pheromones)):
                pheromones[i][j] *= evaporation

        for ant_tour in ant_tours:
            delta_pheromone = 1 / sum(
                graph[i][j] for i, j in zip(ant_tour, ant_tour[1:]))

            for i, j in zip(ant_tour, ant_tour[1:]):
                pheromones[i][j] += delta_pheromone

    best_tour = None
    best_tour_length = float('inf')

    for _ in range(iterations):
        ant_tours = []

        for _ in range(ants):
            unvisited = list(range(num_nodes))
            start = unvisited.pop(random.randint(0, len(unvisited) - 1))
            tour = [start]
            node = start

            while unvisited:
                next_node = select_next_node(node, unvisited, pheromones,
                                             alpha, beta)
                unvisited.remove(next_node)
                tour.append(next_node)
                node = next_node

            tour_length = sum(
                graph[i][j] for i, j in zip(tour, tour[1:] + [start]))

            if tour_length < best_tour_length:
                best_tour = tour
                best_tour_length = tour_length

            ant_tours.append(tour)

        update_pheromones(pheromones, ant_tours, evaporation)

    return best_tour, best_tour_length


# Sample TSP nodes (coordinates)
nodes = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 1)
]

# Create a distance graph and run ACO
graph = generate_graph(nodes)
best_tour, best_tour_length = aco_tsp(graph, ants=10, iterations=100, alpha=1,
                                      beta=2, evaporation=0.9)

print("Best tour:", best_tour)
print("Best tour length:", best_tour_length)


####################################
#### D* Lite algorithm ####
####################################
"""
D* Lite: D* Lite is an incremental heuristic search algorithm designed for 
pathfinding in dynamic environments. It is an extension of the A* algorithm 
and can efficiently adapt its solution as the graph changes, making it 
suitable for applications like robotic navigation.

"""

def neighbors(node, grid):
    x, y = node
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and not grid[nx][ny]:
            yield nx, ny


def heuristic(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def d_star_lite(grid, start, goal):
    costs = {(x, y): float('inf') for x in range(len(grid)) for y in
             range(len(grid[0]))}
    costs[start] = heuristic(start, goal)

    queue = [(costs[start], start)]

    while queue:
        current_cost, node = heapq.heappop(queue)
        x, y = node

        if current_cost != costs[node]:
            continue

        if node == goal:
            return current_cost

        for neighbor in neighbors(node, grid):
            new_cost = current_cost + heuristic(node, neighbor) - heuristic(
                node, goal) + heuristic(neighbor, goal)

            if new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, neighbor))

    return None


# Sample grid (0 = free, 1 = obstacle)
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (4, 4)

# Find the shortest path
shortest_path = d_star_lite(grid, start, goal)

print("Shortest path length from", start, "to", goal, ":", shortest_path)
