import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph representing routers and links
def create_network(num_nodes=10, connection_prob=0.3):
    G = nx.erdos_renyi_graph(num_nodes, connection_prob, directed=False)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(num_nodes, connection_prob, directed=False)
    return G

# Initialize pheromone matrix
def init_pheromone(G, initial_pheromone=1.0):
    pheromone = {}
    for edge in G.edges():
        pheromone[edge] = initial_pheromone
        pheromone[(edge[1], edge[0])] = initial_pheromone  # undirected graph
    return pheromone

# Probability to choose next hop based on pheromone and heuristic (inverse of link cost)
def choose_next_hop(current_node, unvisited, G, pheromone, alpha, beta):
    pheromone_list = []
    heuristic_list = []
    neighbors = list(G.neighbors(current_node))
    feasible_neighbors = [n for n in neighbors if n in unvisited]
    if not feasible_neighbors:
        return None
    
    for neighbor in feasible_neighbors:
        tau = pheromone[(current_node, neighbor)] ** alpha
        eta = (1 / G[current_node][neighbor]['weight']) ** beta
        pheromone_list.append(tau)
        heuristic_list.append(eta)

    probabilities = np.array(pheromone_list) * np.array(heuristic_list)
    probabilities /= probabilities.sum()
    next_node = np.random.choice(feasible_neighbors, p=probabilities)
    return next_node

# Simulate packet routing by ants (packets)
def ant_routing(G, pheromone, alpha, beta, source, destination, packet_drop_rate=0.0):
    path = [source]
    current = source
    visited = set([source])
    while current != destination:
        neighbors = list(G.neighbors(current))
        feasible = [n for n in neighbors if n not in visited]
        if not feasible:
            # Dead end or loop, restart from source or break
            return None  # packet dropped or no route
        next_node = choose_next_hop(current, feasible, G, pheromone, alpha, beta)
        # Simulate packet drop
        if random.random() < packet_drop_rate:
            return None  # packet dropped on this edge
        path.append(next_node)
        visited.add(next_node)
        current = next_node
    return path

# Update pheromones with evaporation and reinforcement
def update_pheromones(G, pheromone, paths, rho, Q=100):
    # Evaporation
    for edge in pheromone:
        pheromone[edge] *= (1 - rho)
        if pheromone[edge] < 0.01:
            pheromone[edge] = 0.01  # minimum pheromone

    # Deposit pheromones for successful paths
    for path in paths:
        if path is None:
            continue
        length = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
        deposit = Q / length
        for i in range(len(path)-1):
            edge = (path[i], path[i+1])
            pheromone[edge] += deposit
            pheromone[(edge[1], edge[0])] += deposit  # undirected

# Run simulation
def run_simulation(num_nodes=10, num_ants=20, alpha=1, beta=2, rho=0.1,
                   packet_drop_rate=0.0, traffic_load=1, iterations=50):
    G = create_network(num_nodes)
    # Assign random weights (delay or cost) to edges
    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(1, 10)

    pheromone = init_pheromone(G)
    source, destination = 0, num_nodes - 1  # fixed source and destination

    best_path = None
    best_length = float('inf')

    for iteration in range(iterations):
        all_paths = []
        for _ in range(num_ants * traffic_load):  # traffic load scales ants count
            path = ant_routing(G, pheromone, alpha, beta, source, destination, packet_drop_rate)
            all_paths.append(path)

        update_pheromones(G, pheromone, all_paths, rho)

        # Track best path found this iteration
        for path in all_paths:
            if path is None:
                continue
            length = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            if length < best_length:
                best_length = length
                best_path = path

        print(f"Iteration {iteration+1} | Best path length: {best_length:.2f} | Packet drops: {sum(p is None for p in all_paths)} / {len(all_paths)} | Traffic load: {traffic_load}")

    # Plot network and best path
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue')
    if best_path:
        path_edges = list(zip(best_path[:-1], best_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)
    plt.title('Network and Best Path Found by ACO')
    plt.show()

    return best_path, best_length

# Example runs:

print("=== Scenario 1: Low packet drop, low traffic ===")
run_simulation(packet_drop_rate=0.05, traffic_load=1)

print("\n=== Scenario 2: Higher packet drop ===")
run_simulation(packet_drop_rate=0.3, traffic_load=1)

print("\n=== Scenario 3: Increased traffic load ===")
run_simulation(packet_drop_rate=0.05, traffic_load=5)
