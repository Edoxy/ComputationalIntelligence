import scipy as scipy
import numpy as np
from random import random, choice, randint
from functools import reduce
from copy import copy
from tqdm.auto import tqdm
from multiprocessing import Pool


class RMHC:
    '''
    Random Mutation Hill Climber
    '''
    def __init__(self, problem_matrix):
        self.problem_matrix = problem_matrix
        self.PROBLEM_SIZE, self.N_SETS = problem_matrix.shape
        self.init_state = [False for _ in range(self.PROBLEM_SIZE)]
        self.fitness_calls = 0

    def tweak(self, state):
        new_state = copy(state)
        n_index = choice([1])
        index = [
            randint(0, self.PROBLEM_SIZE - 1) for _ in range(n_index)
        ]  # changes only the use state of one set
        for i in index:
            new_state[i] = not new_state[i]
        return new_state

    def fitness(self, state):
        self.fitness_calls += 1

        cost = sum(state)
        indexes_set = set()
        overlap_set = set()

        for i, t in enumerate(state):
            if t:
                tmp_set = set(self.problem_matrix.getrow(i).nonzero()[1])
                indexes_set = indexes_set.union(tmp_set)
                overlap_set = overlap_set.intersection(tmp_set)
        n_true = len(indexes_set)
        overlap = len(overlap_set)

        return n_true, -cost, -overlap

    def fitness1(self, state):
        self.fitness_calls += 1
        cost = sum(state)

        n_true = 0
        if np.any(state):
            n_true = np.sum(reduce(np.logical_or, [self.problem_matrix.getrow(i).toarray() for i, t in enumerate(state) if t]))

        valid = 0
        if n_true == self.PROBLEM_SIZE:
            valid = 1

        return valid, n_true / (cost + 1)

    def agent_solver(self, iterations):
        state = self.init_state
        state_fitness = self.fitness(state)
        for it in range(iterations):
            new_state = self.tweak(state)
            fitness_new_state = self.fitness(new_state)

            if fitness_new_state >= state_fitness:
                state = new_state
                state_fitness = fitness_new_state

        return state


class SAHC:
    '''
    Steepest Ascent Hill Climber
    '''
    def __init__(self, problem_matrix, n_samples):
        self.problem_matrix = problem_matrix
        self.PROBLEM_SIZE, self.N_SETS = problem_matrix.shape
        self.init_state = [False for _ in range(self.PROBLEM_SIZE)]
        self.fitness_calls = 0
        self.n_samples = n_samples

    def tweak1(self, state):
        new_state = copy(state)
        n_index = choice([1])
        index = [
            randint(0, self.PROBLEM_SIZE - 1) for _ in range(n_index)
        ]  # changes only the use state of one set
        for i in index:
            new_state[i] = not new_state[i]
        return new_state

    def tweak2(self, state):
        new_state = copy(state)
        n_index = choice([1, 2, 3, 4])
        index = [
            randint(0, self.PROBLEM_SIZE - 1) for _ in range(n_index)
        ]  # changes only the use state of one set
        for i in index:
            new_state[i] = not new_state[i]
        return new_state

    def fitness(self, state):
        self.fitness_calls += 1
        cost = sum(state)

        indexes_set = set()
        overlap_set = set()
        for i, t in enumerate(state):
            if t:
                tmp_set = set(self.problem_matrix.getrow(i).nonzero()[1])
                indexes_set = indexes_set.union(tmp_set)
                overlap_set = overlap_set.intersection(tmp_set)
        n_true = len(indexes_set)
        overlap = len(overlap_set)

        return n_true, -cost, -overlap

    def fitness1(self, state):
        self.fitness_calls += 1
        cost = sum(state)
        indexes_set = set()
        overlap_set = set()

        for i, t in enumerate(state):
            if t:
                tmp_set = set(self.problem_matrix.getrow(i).nonzero()[1])
                indexes_set = indexes_set.union(tmp_set)
        n_true = len(indexes_set)

        valid = 0
        if n_true == self.PROBLEM_SIZE:
            valid = 1

        return valid, n_true / (cost + 1)

    def agent_solver(self, iterations):
        state = self.init_state
        state_fitness = self.fitness1(state)

        for it in range(iterations):
            new_state = self.tweak1(state)
            new_state_fitness = self.fitness1(new_state)

            for s in range(self.n_samples):
                tmp_state = self.tweak2(new_state)
                tmp_fitness = self.fitness1(tmp_state)

                if tmp_fitness >= new_state_fitness:
                    new_state = tmp_state
                    new_state_fitness = tmp_fitness

            if new_state_fitness >= state_fitness:
                state = new_state
                state_fitness = new_state_fitness

        return state


class TabuSearch:
    '''
    Tabu Search addition to the SAHC class
    '''
    def __init__(self, problem_matrix, n_samples, auto_stop):
        self.problem_matrix = problem_matrix
        self.PROBLEM_SIZE, self.N_SETS = problem_matrix.shape
        self.init_state = tuple([False for _ in range(self.PROBLEM_SIZE)])
        self.fitness_calls = 0
        self.n_samples = n_samples
        self.tabu = set()
        self.auto_stop = auto_stop

    def count_ones(self, state):
        '''
        
        '''
        indexes_set = set()
        for i, t in enumerate(state):
            if t:
                tmp_set = set(self.problem_matrix.getrow(i).nonzero()[1])
                indexes_set = indexes_set.union(tmp_set)
        return len(indexes_set)

    def tweak1(self, state):
        new_state = list(state)
        n_index = choice([1])
        index = [
            randint(0, self.PROBLEM_SIZE - 1) for _ in range(n_index)
        ]  # changes only the use state of one set
        for i in index:
            new_state[i] = not new_state[i]
        new_state = tuple(new_state)

        return new_state

    def tweak2(self, state):
        new_state = list(state)
        n_index = choice([1,2])
        index = [
            randint(0, self.PROBLEM_SIZE - 1) for _ in range(n_index)
        ]  # changes only the use state of one set
        for i in index:
            new_state[i] = not new_state[i]
        new_state = tuple(new_state)

        return new_state

    def fitness(self, state):
        self.fitness_calls += 1
        cost = sum(state)

        indexes_set = set()
        overlap_set = set()
        for i, t in enumerate(state):
            if t:
                tmp_set = set(self.problem_matrix.getrow(i).nonzero()[1])
                indexes_set = indexes_set.union(tmp_set)
                overlap_set = overlap_set.intersection(tmp_set)
        n_true = len(indexes_set)
        overlap = len(overlap_set)

        return n_true, -cost, -overlap

    def fitness1(self, state):
        self.fitness_calls += 1
        cost = sum(state)
        n_true = 0
        if np.any(state):
            n_true = np.sum(reduce(np.logical_or, [self.problem_matrix.getrow(i).toarray() for i, t in enumerate(state) if t]))

        return n_true, -cost

    def agent_solver(self, iterations):
        state = self.init_state
        state_fitness = self.fitness1(state)

        for _ in range(iterations):
            new_state = self.tweak1(state)
            if new_state in self.tabu:
                continue
            new_state_fitness = self.fitness1(new_state)

            for _ in range(self.n_samples):
                tmp_state = self.tweak2(state)
                if tmp_state in self.tabu:
                    continue
                tmp_fitness = self.fitness1(tmp_state)

                if tmp_fitness >= new_state_fitness:
                    self.tabu.add(new_state_fitness)

                    new_state = tmp_state
                    new_state_fitness = tmp_fitness

                else:
                    self.tabu.add(tmp_state)

            if new_state_fitness >= state_fitness:
                self.tabu.add(state)
                state = new_state
                state_fitness = new_state_fitness
            else:
                self.tabu.add(new_state)

        return state
    
    def agent_solver_autostop(self, iterations):
        state = copy(self.init_state)
        state_fitness = self.fitness1(state)
        
        no_progress_counter = 0

        with tqdm(total=iterations) as pbar:
            for _ in range(iterations):
                pbar.update(1)
                new_state = self.tweak1(state)
                if new_state in self.tabu:
                    continue
                new_state_fitness = self.fitness1(new_state)

                for s in range(self.n_samples):
                    tmp_state = self.tweak2(state)
                    if tmp_state in self.tabu:
                        continue
                    tmp_fitness = self.fitness1(tmp_state)

                    if tmp_fitness >= new_state_fitness:
                        self.tabu.add(new_state_fitness)

                        new_state = tmp_state
                        new_state_fitness = tmp_fitness

                    else:
                        self.tabu.add(tmp_state)

                if new_state_fitness >= state_fitness:
                    self.tabu.add(state)
                    state = new_state
                    state_fitness = new_state_fitness

                    no_progress_counter = 0
                else:
                    if state_fitness[0] == self.PROBLEM_SIZE:
                        no_progress_counter +=1
                    self.tabu.add(new_state)

                    if no_progress_counter >= self.auto_stop:
                        break
            pbar.close()

        return state, state_fitness
    

class ILS(TabuSearch):
    '''
    Iterated Local Search based on the TabuSearch Class
    Random restarts are executed when the Tabu Search returns a local maximum
    '''
    def __init__(self, problem_matrix, n_samples, auto_stop, restart):
        super().__init__(problem_matrix, n_samples, auto_stop)
        self.global_max_state = None
        self.global_max_fitness = (-1e14, -1e14)
        self.n_restart = restart

    def agent_solver(self, iterations):

        for i in range(self.n_restart):
            local_best, local_best_fitness = super().agent_solver_autostop(iterations)
            if local_best_fitness >= self.global_max_fitness:
                self.global_max_state = local_best
                self.global_max_fitness = local_best_fitness
            self.init_state = tuple([False for _ in range(self.PROBLEM_SIZE)])
        return self.global_max_state
    

class ILS_parallel(TabuSearch):
    '''
    Multiprocessing implementation od the Iterated Local Search class (experimental)
    '''
    def __init__(self, problem_matrix, n_samples, auto_stop, restart):
        super().__init__(problem_matrix, n_samples, auto_stop)
        self.global_max_state = None
        self.global_max_fitness = (-1e14, -1e14)
        self.n_restart = restart

    def agent_solver(self, iterations):
        num_processors = self.n_restart
        pool = Pool(processes=num_processors)
        output = pool.map(super().agent_solver_autostop, [iterations for _ in range(num_processors)])
        pool.close()
        pool.join()
        
        output.sort(key = lambda x: x[1], reverse=True)
        return output[0][0]

        