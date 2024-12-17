import numpy as np
import random


def create_network(nodes, target_conns):
    connections = np.zeros((nodes, nodes), dtype=int)
    loads = np.random.uniform(0.1, 1.0, nodes)

    for i in range(1, nodes):
        j = random.randint(0, i - 1)
        connections[i][j] = connections[j][i] = 1

    while np.sum(connections) / 2 < target_conns * nodes:
        i = random.randint(0, nodes - 1)
        j = random.randint(0, nodes - 1)
        if i != j and connections[i][j] == 0:
            connections[i][j] = connections[j][i] = 1

    return {'connections': connections, 'loads': loads}


def calc_fitness(network, target_conns):
    connections = network['connections']
    loads = network['loads']

    penalty = 0
    for i in range(len(connections)):
        if sum(connections[i]) == 0:
            penalty += 1000

    conn_counts = np.sum(connections, axis=1)
    conn_var = np.var(conn_counts)
    conn_dev = abs(np.mean(conn_counts) - target_conns)
    conn_penalty = conn_var * 50 + conn_dev * 20

    load_penalty = np.var(loads) * 100

    total = np.sum(connections) / 2
    excess = max(0, total - target_conns * len(connections))
    excess_penalty = excess * 10

    return penalty + conn_penalty + load_penalty + excess_penalty


def tournament_select(population, fitnesses):
    idx = random.sample(range(len(population)), 3)
    return population[min(idx, key=lambda x: fitnesses[x])]


def make_child(parent1, parent2):
    nodes = len(parent1['connections'])
    child = {
        'connections': np.zeros((nodes, nodes), dtype=int),
        'loads': np.zeros(nodes)
    }

    for i in range(nodes):
        for j in range(i + 1, nodes):
            val = parent1['connections'][i][j] if random.random() < 0.5 else parent2['connections'][i][j]
            child['connections'][i][j] = child['connections'][j][i] = val

        mix = random.random()
        child['loads'][i] = mix * parent1['loads'][i] + (1 - mix) * parent2['loads'][i]

    return child


def mutate_network(network, mut_rate, target_conns):
    nodes = len(network['connections'])
    conn_counts = np.sum(network['connections'], axis=1)

    for i in range(nodes):
        if random.random() < mut_rate:
            max_node = np.argmax(conn_counts)
            min_node = np.argmin(conn_counts)

            if conn_counts[min_node] < target_conns:
                possible = [j for j in range(nodes)
                            if j != min_node and network['connections'][min_node][j] == 0]
                if possible:
                    j = random.choice(possible)
                    network['connections'][min_node][j] = network['connections'][j][min_node] = 1
                    conn_counts[min_node] += 1
                    conn_counts[j] += 1

            elif conn_counts[max_node] > target_conns:
                connected = [j for j in range(nodes)
                             if network['connections'][max_node][j] == 1]
                if len(connected) > 1:
                    j = random.choice(connected)
                    network['connections'][max_node][j] = network['connections'][j][max_node] = 0
                    conn_counts[max_node] -= 1
                    conn_counts[j] -= 1

        if random.random() < mut_rate:
            network['loads'][i] *= random.uniform(0.8, 1.2)
            network['loads'][i] = max(0.1, min(1.0, network['loads'][i]))


def run_evolution(pop_size, nodes, target_conns, mut_rate, elite_size, max_gens):
    population = [create_network(nodes, target_conns) for _ in range(pop_size)]
    fitnesses = [calc_fitness(net, target_conns) for net in population]
    history = []
    no_improvement = 0
    best_fit = float('inf')

    for gen in range(max_gens):
        sorted_idx = np.argsort(fitnesses)
        population = [population[i] for i in sorted_idx]
        fitnesses = [fitnesses[i] for i in sorted_idx]

        history.append(fitnesses[0])

        if fitnesses[0] < best_fit:
            best_fit = fitnesses[0]
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= 30:
            break

        new_pop = population[:elite_size]
        new_fits = fitnesses[:elite_size]

        while len(new_pop) < pop_size:
            parent1 = tournament_select(population, fitnesses)
            parent2 = tournament_select(population, fitnesses)
            child = make_child(parent1, parent2)
            mutate_network(child, mut_rate, target_conns)
            new_pop.append(child)
            new_fits.append(calc_fitness(child, target_conns))

        population = new_pop
        fitnesses = new_fits

    best_idx = np.argmin(fitnesses)
    return population[best_idx], history


pop_size = 100
num_nodes = 10
target_conns = 3
mut_rate = 0.15
elite = 2
max_gens = 200

best, history = run_evolution(pop_size, num_nodes, target_conns, mut_rate, elite, max_gens)
print(f"Best fitness: {calc_fitness(best, target_conns)}")
print("Connections:")
print(best['connections'])
print("Node loads:")
print(best['loads'])

conn_counts = np.sum(best['connections'], axis=1)

for i, count in enumerate(conn_counts):
    connected = [j for j in range(num_nodes) if best['connections'][i][j] == 1]
    print(f"Node {i}: {count} connections with {connected}")