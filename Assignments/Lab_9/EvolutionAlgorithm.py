"""
Copyright **`(c)`** 2023 Edoardo Vay  `<vay.edoardo@gmail.com>`
<https://github.com/Edoxy>
"""

import random
from random import choices, randint
import numpy as np
from copy import deepcopy
import time
import heapq
from tqdm.auto import tqdm
from multiprocessing import Pool
from typing import Callable, List, Tuple


def xover_bisection(
    parent1: Tuple[float, List[bool]], parent2: Tuple[float, List[bool]]
) -> np.array:
    """
    Function executing a simple crossover between 2 parents in the middle point\n
    return: genome of the new individual
    """

    _, parent1 = parent1
    _, parent2 = parent2

    if not len(parent1) == len(parent2):
        print("Parent lenght must be the same")
        return None

    xover_point = len(parent1) // 2

    newborn = np.concatenate((parent1[:xover_point], parent2[xover_point:]))
    return newborn


def xover_multiple_point(
    parent1: Tuple[float, List[bool]], parent2: Tuple[float, List[bool]]
) -> np.array:
    """
    Function executing a crossover between 2 parents in in 4 randomly sampled points\n
    return: genome of the new individual
    """
    _, parent1 = parent1
    _, parent2 = parent2

    if not len(parent1) == len(parent2):
        print("Parent lenght must be the same")
        print(len(parent1), len(parent2))
        return None
    p = np.array([random.randint(0, len(parent1) - 1) for i in range(4)])
    p = np.sort(p)
    newborn = np.concatenate(
        (
            parent1[0 : p[0]],
            parent2[p[0] : p[1]],
            parent1[p[1] : p[2]],
            parent2[p[2] : p[3]],
            parent1[p[3] :],
        )
    )
    return newborn


def xover_uniform(
    parent1: Tuple[float, List[bool]], parent2: Tuple[float, List[bool]]
) -> np.array:
    """
    Uniform Crossover function
    """

    _, parent1 = parent1
    _, parent2 = parent2

    if not len(parent1) == len(parent2):
        print("Parent lenght must be the same")
        return None

    random_mask = np.array(choices([0, 1], k=len(parent1)), dtype=bool)

    masked_parent1 = np.logical_and(random_mask, parent1)
    masked_parent2 = np.logical_and(~random_mask, parent2)

    newborn = np.logical_or(masked_parent1, masked_parent2)
    return newborn.astype(int)


def xover_random(
    parent1: Tuple[float, List[bool]], parent2: Tuple[float, List[bool]]
) -> np.array:
    """
    Crossover function that favours a parent based on the fitness
    """

    fit_parent1, parent1 = parent1
    fit_parent2, parent2 = parent2

    p = fit_parent1 / (fit_parent1 + fit_parent2)
    if not len(parent1) == len(parent2):
        print("Parent lenght must be the same")
        return None

    random_mask = np.array(
        choices([0, 1], weights=[1 - p, p], k=len(parent1)), dtype=bool
    )

    masked_parent1 = np.logical_and(random_mask, parent1)
    masked_parent2 = np.logical_and(~random_mask, parent2)

    newborn = np.logical_or(masked_parent1, masked_parent2)
    return newborn.astype(int)


def xover_or(
    parent1: Tuple[float, List[bool]], parent2: Tuple[float, List[bool]]
) -> np.array:
    """
    test
    """

    _, parent1 = parent1
    _, parent2 = parent2

    if not len(parent1) == len(parent2):
        print("Parent lenght must be the same")
        return None
    newborn = np.logical_or(parent1, parent2)

    return newborn.astype(int)


def random_change(parent: Tuple[float, List[bool]]) -> np.array:
    """
    Genetic Operator that mutates a parent
    """
    _, parent = parent
    index = randint(0, len(parent) - 1)
    newborn = deepcopy(parent)
    newborn[index] = not newborn[index]
    return newborn.astype(int)


def tournament(population_fitness: List[float], tournament_size: int) -> np.array:
    """
    !! THIS VERSION only works with the first iteration of EA -> USE INSTEAD tournament_heap(), or tournament_opt() with the other version of EA (EA_heap, EA_Extintions and EA_Islands)
    Function that choses a parent with tournament selection
    tournament_size: (int) size of the tournament
    return: index of the parent
    """
    candidates = [
        randint(0, len(population_fitness) - 1) for _ in range(tournament_size)
    ]
    return candidates[np.argmax(population_fitness[candidates])]


def EA(
    mu: int,
    lamda: int,
    generations: int,
    tournament_size: int,
    fitness: Callable,
    mutation_func: Callable,
    xover_func: Callable,
    parent_selection_func: Callable,
) -> Tuple[float, np.array]:
    """
    Function execution a Genetic Algorithm. First version created: it has poor performance because of the sorting used

    mu: (int) is the population size

    lamda: (int) is the number of parent that are selected each generation

    generations: (int) is the number of generation that the algorithm will run

    fitness: function that evaluate the individual

    mutation_func: genetic operator that, given a single parent makes a mutation

    xover_func: genetic operator that, given 2 parents, makes an offspring

    parent_selection_func:

    return: (tuple) fitness, best individual
    """

    MUTATION_PROB = 0.1
    # matrix where each row is an individual of the population
    population = np.array([choices([0, 1], k=1000) for _ in range(mu)])
    # vector with fitness for each individual in the population
    population_fitness = np.array([fitness(population[i]) for i in range(mu)])

    offsprings = list()
    offsprings_fitness = list()
    # Generation Cycle
    time_generating_offspring = 0
    time_sorting = 0
    for gen in range(generations):
        # Offspring Cycle
        time_generating_offspring = 0

        start_leg = time.time()

        for newborn in range(lamda):
            if random.random() < MUTATION_PROB:
                # do mutation
                # extract the index of the parent
                parent_index = parent_selection_func(
                    population_fitness, tournament_size=tournament_size
                )
                # create a new offspring
                newborn = mutation_func(population[parent_index])
                offsprings.append(newborn)
                offsprings_fitness.append(fitness(newborn))

            else:
                # 2 parents and xover
                parent_index1 = parent_selection_func(
                    population_fitness, tournament_size=tournament_size
                )
                parent_index2 = parent_selection_func(
                    population_fitness, tournament_size=tournament_size
                )

                newborn = xover_func(
                    population[parent_index1], population[parent_index2]
                )
                offsprings.append(newborn)
                offsprings_fitness.append(fitness(newborn))

        stop_leg = time.time()
        time_generating_offspring += stop_leg - start_leg

        start_sorting = time.time()

        population = np.concatenate((population, offsprings), axis=0)
        population_fitness = np.concatenate(
            (population_fitness, offsprings_fitness), axis=0
        )

        sorted_idexes = np.argsort(-population_fitness)

        population = population[sorted_idexes[0:mu]]
        population_fitness = population_fitness[sorted_idexes[0:mu]]

        stop_sorting = time.time()

        time_sorting += stop_sorting - start_sorting

    print(f"Time spent generating {time_generating_offspring:.2}")
    print(f"Time spent sorting {time_sorting:.2}")
    return population[-1], population_fitness[-1]


def tournament_heap(
    population: List[Tuple[float, List[bool]]], tournament_size: int
) -> np.array:
    """
    Function that choses a parent with tournament selection
    tournament_size: (int) size of the tournament
    return: index of the parent
    """
    candidates = [randint(0, len(population) - 1) for _ in range(tournament_size)]

    return candidates[np.argmax([population[i][0] for i in candidates])]


def tournament_opt(
    population: List[Tuple[float, List[bool]]], tournament_size: int
) -> np.array:
    """
    Function that choses a parent with tournament selection
    tournament_size: (int) size of the tournament
    return: index of the parent
    """
    pop_size = len(population)

    winner_index = randint(0, pop_size - 1)
    winner_fitness = population[winner_index][0]

    for _ in range(tournament_size - 1):
        candidate = randint(0, pop_size - 1)
        candidate_fitness = population[candidate][0]

        if candidate_fitness > winner_fitness:
            winner_fitness = candidate_fitness
            winner_index = candidate

    return winner_index


def EA_heap(
    mu: int,
    lamda: int,
    generations: int,
    tournament_size: int,
    MUTATION_PROB: float,
    fitness: Callable[[List[bool]], float],
    mutation_func: Callable[
        [Tuple[float, List[bool]], Tuple[float, List[bool]]], List[bool]
    ],
    xover_func: Callable[[Tuple[float, List[bool]]], List[bool]],
    parent_selection_func: Callable[[List[Tuple[float, List[bool]]]], List[bool]],
) -> Tuple[float, List[bool]]:
    """
    Function execution a Genetic Algorithm that uses and Heapq to improve the efficiency.\n
    mu: (int) > Is the population size\n
    lamda: (int) > Is the number of parent that are selected each generation\n
    generations: (int) > Is the number of generation that the algorithm will run\n
    tournament_size: (int) > Maximum number of individual partecipating in the tournament for the parent selection
    MUTATION_PROB: (float) > Probability that a newborn is generated from a mutation instead of a crossover\n
    fitness: (function) Function that evaluate the individual\n
    mutation_func: (function) > Genetic operator that, given a single parent makes a mutation\n
    xover_func: (function) > Genetic operator that, given 2 parents, makes an offspring\n
    parent_selection_func: (function) >  Function that, given the population and its fitness, return the parent selected\n
    return: (tuple) > (fitness of the best individual, best individual)\n
    """
    # matrix where each row is an individual of the population
    population = []
    for _ in range(mu):
        indiv = np.array(
            choices(
                [0, 1], weights=[random.randint(1, mu), random.randint(1, mu)], k=1000
            ),
            dtype=int,
        )
        indiv_fitness = fitness(indiv)
        heapq.heappush(population, (indiv_fitness + 1e-6 * random.random(), indiv))

    # Generation Cycle
    time_generating_offspring = 0
    time_sorting = 0
    # for gen in range(generations):
    for step in tqdm(range(generations)):
        # Offspring Cycle

        start_leg = time.time()

        for newborn in range(lamda):
            if random.random() < MUTATION_PROB:
                # do mutation
                # extract the index of the parent
                parent_index = parent_selection_func(
                    population, tournament_size=tournament_size
                )
                # create a new offspring
                parent = population[parent_index]
                newborn = mutation_func(parent)
                newborn_fitness = fitness(newborn)

                start_sorting = time.time()
                heapq.heappushpop(
                    population, (newborn_fitness + 1e-6 * random.random(), newborn)
                )
                stop_sorting = time.time()

            else:
                # 2 parents and xover
                parent_index1 = parent_selection_func(
                    population, tournament_size=tournament_size
                )
                parent_index2 = parent_selection_func(
                    population, tournament_size=tournament_size
                )
                # _, parent1 = population[parent_index1]
                # _, parent2 = population[parent_index2]
                parent1 = population[parent_index1]
                parent2 = population[parent_index2]

                newborn = xover_func(parent1, parent2)
                newborn_fitness = fitness(newborn)

                start_sorting = time.time()
                heapq.heappushpop(
                    population, (newborn_fitness + 1e-6 * random.random(), newborn)
                )
                stop_sorting = time.time()

            time_sorting += stop_sorting - start_sorting

        stop_leg = time.time()
        time_generating_offspring += stop_leg - start_leg

    # print(f"Time spent generating {time_generating_offspring:.3} seconds")
    #  print(f"Time spent sorting {time_sorting:.3} seconds")

    return heapq.nlargest(1, population)[0]


def EA_Extintions(
    mu: int,
    lamda: int,
    generations: int,
    tournament_size: int,
    MUTATION_PROB: float,
    n_extintions: int,
    fitness: Callable[[List[bool]], float],
    mutation_func: Callable[
        [Tuple[float, List[bool]], Tuple[float, List[bool]]], List[bool]
    ],
    xover_func: Callable[[Tuple[float, List[bool]]], List[bool]],
    parent_selection_func: Callable[[List[Tuple[float, List[bool]]]], List[bool]],
) -> Tuple[float, List[bool]]:
    """
    Function execution a Genetic Algorithm that uses Extintions. Extintion are executed at regular times based on how many generation are setted\n
    mu: (int) > Is the population size\n
    lamda: (int) > Is the number of parent that are selected each generation\n
    generations: (int) > Is the number of generation that the algorithm will run\n
    tournament_size: (int) > Maximum number of individual partecipating in the tournament for the parent selection
    MUTATION_PROB: (float) > Probability that a newborn is generated from a mutation instead of a crossover\n
    n_extintions: (int) > Number of mass-extintion to be executed
    fitness: (function) Function that evaluate the individual\n
    mutation_func: (function) > Genetic operator that, given a single parent makes a mutation\n
    xover_func: (function) > Genetic operator that, given 2 parents, makes an offspring\n
    parent_selection_func: (function) >  Function that, given the population and its fitness, return the parent selected\n
    return: (tuple) > (fitness of the best individual, best individual)\n
    """
    # how many generation before an Extintion
    EXTINTION_TIME = generations // n_extintions
    #
    population = []
    # create the first individual
    first = np.array(
        choices([0, 1], weights=[random.randint(1, mu), random.randint(1, mu)], k=1000),
        dtype=int,
    )
    first_fitness = fitness(first)
    heapq.heappush(population, (first_fitness + 1e-6 * random.random(), first))

    # Generation Cycle
    time_generating_offspring = 0
    time_sorting = 0
    # for gen in range(generations):
    for step in tqdm(range(generations)):
        # Extintion check
        if step % EXTINTION_TIME == 0:
            # kill every one but one
            survivor = heapq.nlargest(1, population)[0]
            population = []
            heapq.heappush(population, survivor)

            for _ in range(mu - 1):
                new_indv = np.array(
                    choices(
                        [0, 1],
                        weights=[random.randint(1, mu), random.randint(1, mu)],
                        k=1000,
                    ),
                    dtype=int,
                )
                new_indv_fitness = fitness(new_indv)
                heapq.heappush(
                    population, (new_indv_fitness + 1e-6 * random.random(), new_indv)
                )

        start_leg = time.time()
        # Offspring Cycle
        for newborn in range(lamda):
            if random.random() < MUTATION_PROB:
                # do mutation
                # extract the index of the parent
                parent_index = parent_selection_func(
                    population, tournament_size=tournament_size
                )
                # create a new offspring
                parent = population[parent_index]
                newborn = mutation_func(parent)
                newborn_fitness = fitness(newborn)

                start_sorting = time.time()
                heapq.heappushpop(
                    population, (newborn_fitness + 1e-6 * random.random(), newborn)
                )
                stop_sorting = time.time()

            else:
                # 2 parents and xover
                parent_index1 = parent_selection_func(
                    population, tournament_size=tournament_size
                )
                parent_index2 = parent_selection_func(
                    population, tournament_size=tournament_size
                )

                parent1 = population[parent_index1]
                parent2 = population[parent_index2]

                newborn = xover_func(parent1, parent2)
                newborn_fitness = fitness(newborn)

                start_sorting = time.time()
                heapq.heappushpop(
                    population, (newborn_fitness + 1e-6 * random.random(), newborn)
                )
                stop_sorting = time.time()

            time_sorting += stop_sorting - start_sorting

        stop_leg = time.time()
        time_generating_offspring += stop_leg - start_leg

    # print(f"Time spent generating {time_generating_offspring:.3} seconds")
    # print(f"Time spent sorting {time_sorting:.3} seconds")

    return heapq.nlargest(1, population)[0]


def EA_Islands(
    mu: int,
    lamda: int,
    generations: int,
    tournament_size: int,
    n_islands: int,
    MUTATION_PROB: float,
    fitness: Callable[[List[bool]], float],
    mutation_func: Callable[
        [Tuple[float, List[bool]], Tuple[float, List[bool]]], List[bool]
    ],
    xover_func: Callable[[Tuple[float, List[bool]]], List[bool]],
    parent_selection_func: Callable[[List[Tuple[float, List[bool]]]], List[bool]],
) -> Tuple[float, List[bool]]:
    """
    Function execution a Genetic Algorithm that uses Niching to preserve diverity. The different islands are executed in parallel and with a tournament size randomly choosen to simulate different environments\n
    mu: (int) > Is the population size\n
    lamda: (int) > Is the number of parent that are selected each generation\n
    generations: (int) > Is the number of generation that the algorithm will run\n
    tournament_size: (int) > Maximum number of individual partecipating in the tournament for the parent selection
    n_islands: (int) > Number of indipendent islands created\n
    MUTATION_PROB: (float) > Probability that a newborn is generated from a mutation instead of a crossover\n
    fitness: (function) Function that evaluate the individual\n
    mutation_func: (function) > Genetic operator that, given a single parent makes a mutation\n
    xover_func: (function) > Genetic operator that, given 2 parents, makes an offspring\n
    parent_selection_func: (function) >  Function that, given the population and its fitness, return the parent selected\n
    return: (tuple) > (fitness of the best individual, best individual)\n
    """
    # ---------------------------------------------------------------------------------
    # Parallel Islands
    # ---------------------------------------------------------------------------------

    # Number of processors that will be used in parallel
    N_PROCESSORS = 7
    pool = Pool(processes=N_PROCESSORS)
    # Execute the islands in parallel
    population = pool.starmap(
        EA_heap,
        [
            [
                mu,
                lamda,
                generations,
                random.randint(
                    1, tournament_size
                ),  # random tournament size is chosen to simulate diffent environments with different selective Pressure
                MUTATION_PROB,
                fitness,
                mutation_func,
                xover_func,
                parent_selection_func,
            ]
            for _ in range(n_islands)
        ],
    )

    # ---------------------------------------------------------------------------------
    # Union of the champion of each island
    # ---------------------------------------------------------------------------------

    # The champion of each island is inserted in a new heapq to start a new population
    heapq.heapify(population)
    # populating the remaning places with random individual
    for _ in range(mu - n_islands):
        indiv = np.array(
            choices(
                [0, 1], weights=[random.randint(1, mu), random.randint(1, mu)], k=1000
            ),
            dtype=int,
        )
        indiv_fitness = fitness(indiv)
        heapq.heappush(population, (indiv_fitness + 1e-6 * random.random(), indiv))

    # Generation Cycle
    time_generating_offspring = 0
    time_sorting = 0
    # for gen in range(generations):
    for step in tqdm(range(generations)):
        # Offspring Cycle

        start_leg = time.time()

        for newborn in range(lamda):
            if random.random() < MUTATION_PROB:
                # do mutation
                # extract the index of the parent
                parent_index = parent_selection_func(
                    population, tournament_size=tournament_size
                )
                # create a new offspring
                parent = population[parent_index]
                newborn = mutation_func(parent)
                newborn_fitness = fitness(newborn)

                start_sorting = time.time()
                heapq.heappushpop(
                    population, (newborn_fitness + 1e-6 * random.random(), newborn)
                )
                stop_sorting = time.time()

            else:
                # 2 parents and xover
                parent_index1 = parent_selection_func(
                    population, tournament_size=tournament_size
                )
                parent_index2 = parent_selection_func(
                    population, tournament_size=tournament_size
                )
                # _, parent1 = population[parent_index1]
                # _, parent2 = population[parent_index2]
                parent1 = population[parent_index1]
                parent2 = population[parent_index2]

                newborn = xover_func(parent1, parent2)
                newborn_fitness = fitness(newborn)

                start_sorting = time.time()
                heapq.heappushpop(
                    population, (newborn_fitness + 1e-6 * random.random(), newborn)
                )
                stop_sorting = time.time()

            time_sorting += stop_sorting - start_sorting

        stop_leg = time.time()
        time_generating_offspring += stop_leg - start_leg

    # print(f"Time spent generating {time_generating_offspring:.3} seconds")
    # print(f"Time spent sorting {time_sorting:.3} seconds")

    return heapq.nlargest(1, population)[0]
