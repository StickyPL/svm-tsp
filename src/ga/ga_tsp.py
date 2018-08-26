import math
import numpy as np
import os
import pandas as pd
import random
import sys

from ga.io_helper import read_tsp, normalize

TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.01


def usage(script):
    print(script+' <tsp filename>.tsp')


def __main__():
    args = sys.argv
    if len(args) < 2:
        usage(args[0])
        exit(1)
    filename = os.path.abspath(args[1])
    print("Loading "+filename)
    problem = read_tsp(filename)
    ga(problem)
    pass


def ga(problem):
    """Solve the TSP using a Genetic Algorithm"""

    # Obtain the normalized set of cities (w/ coord in [0,1])
    cities = problem.copy()

    cities[['x', 'y']] = normalize(cities[['x', 'y']])

    dim = cities.shape[0]
    print(dim)

    pop = list(generate_population(cities, 50))

    print("Initial distance:")
    print(get_fittest(pop).iloc[-1]['dist'])

    # Evolve population for 100 generations
    for i in range(100):
        if i % 10 == 0:
            print("Generation #{}".format(i))
        pop = evolve_population(pop, True)

    print("Finished")
    f = get_fittest(pop)
    dist = f.iloc[-1]['dist']
    fitness = f.iloc[-1]['fitness']
    print("Final distance:")
    print(dist)
    print("Solution:")
    print(f[['city', 'x', 'y']])


def generate_population(cities, size):
    return (sample_tour(cities) for _ in range(size))


def evolve_population(pop, elitism):
    new_pop = []
    size = len(pop)
    elitism_offset = 0

    if elitism:
        new_pop.append(get_fittest(pop))
        elitism_offset = 1

    for i in range(size - elitism_offset):
        parent1 = tournament_selection(pop)
        parent2 = tournament_selection(pop)

        if parent1 is None or parent2 is None:
            print("error")

        child = crossover(parent1, parent2)
        new_pop.append(child)

    # Mutate new population
    for i in range(elitism_offset, size):
        mutant = mutate(new_pop[i])
        new_pop[i] = mutant
        dist = calc_distance(new_pop[i])
        new_pop[i]['dist'] = dist
        new_pop[i]['fitness'] = calc_fitness(dist)

    return new_pop


def mutate(tour):
    cities_to_change = tour.sample(frac=MUTATION_RATE)
    for index1 in cities_to_change.index:
        # index1 = city.index
        index2 = tour[tour.index != index1].sample().index
        index1 = tour[tour.index == index1].index
        # swap cities with index1 and index2
        tour.iloc[index1], tour.iloc[index2] = tour.iloc[index2].copy(), tour.iloc[index1].copy()
    return tour


def tournament_selection(pop):
    tournament = random.sample(pop, TOURNAMENT_SIZE)
    return get_fittest(tournament)


def crossover(parent1, parent2):
    frac = np.random.random()
    child = parent1.sample(frac=frac).sort_index()
    excluded_cities = parent2.city.isin(child.city)
    new_index = [i for i in range(parent2.shape[0]) if i not in child.index]
    child = pd.concat([child, parent2[~excluded_cities].reindex(new_index)]).sort_index()
    # print(child)
    return child


def sample_tour(cities):
    tour = cities.sample(frac=1).reset_index(drop=True)
    tour['dist'] = calc_distance(tour)
    tour['fitness'] = calc_fitness(tour['dist'])
    return tour


def calc_distance(tour):
    distance = 0.0
    current = tour.iloc[0]
    size = tour.shape[0]
    for i in range(1, size):
        city = tour.iloc[i]
        distance += math.sqrt((city.x - current.x)**2 + (city.y-current.y)**2)
        current = city
    return distance


def calc_fitness(distance):
    return 1.0/distance


def get_fittest(population):
    max_fit = population[0].iloc[-1]['fitness']
    fittest = population[0]
    for i in range(1, len(population)):
        tour = population[i]
        fit = tour.iloc[-1]['fitness']
        if max_fit < fit:
            max_fit = fit
            fittest = tour
    print("Best distance so far: {}".format(fittest.iloc[-1]['dist']))
    return fittest


__main__()
