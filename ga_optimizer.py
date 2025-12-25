import random

def genetic_optimize(features, target_sum):
    population = []
    k = len(features)

    for _ in range(50):
        individual = [random.uniform(0.5, 1.5) for _ in range(k)]
        population.append(individual)

    def fitness(ind):
        weighted = sum(ind[i] * features[i] for i in range(k))
        return abs(target_sum - weighted)

    for _ in range(30):
        population.sort(key=fitness)
        parents = population[:10]

        children = parents.copy()
        while len(children) < 50:
            a, b = random.sample(parents, 2)
            point = random.randint(1, k - 1)
            child = a[:point] + b[point:]
            if random.random() < 0.1:
                m = random.randint(0, k - 1)
                child[m] *= random.uniform(0.8, 1.2)
            children.append(child)

        population = children

    best = population[0]
    optimized = [best[i] * features[i] for i in range(k)]
    return optimized
