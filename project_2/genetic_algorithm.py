import random

import numpy as np


class Individual(object):
    def __init__(self, chromossome_mutation_rate=0.):
        """
        Generate a new individual
        """
        self.chromossome_mutation_rate = chromossome_mutation_rate
        pass

    def fitness(self):
        """
        Calculate self fitness value
        :return: individual fitness
        """
        pass

    def clone(self):
        """
        Creates a copy of self
        :return: a copy of self
        """
        pass

    def breed(self, other):
        """
        Mutates self based on another individual (becomes its own offspring)
        :param other: mate partner
        """
        pass

    def mutate(self):
        """
        Mutates self
        """
        pass


class Population(object):
    def __init__(self, source, size=None, mutation_rate=0., crossover_rate=0.):
        if isinstance(source, Population):
            self.population = [ind.clone() for ind in source.population]
            self.size = source.size
            self.crossover_rate = source.crossover_rate
            self.mutation_rate = source.mutation_rate
        else:
            self.size = size
            self.mutation_rate = mutation_rate
            self.crossover_rate = crossover_rate
            if isinstance(source, list):
                self.population = source
            else:
                self.population = [source() for _ in range(size)]

    def select_breeding_pool(self):
        pass

    def clone(self):
        pass

    def rank(self):
        return np.array(list(map(lambda ind: ind.fitness(), self.population))).max()

    def generate(self):
        offspring = self.clone()
        parents = offspring.select_breeding_pool()
        for i in range(1, len(parents) - 1, 2):
            if random.random() < offspring.crossover_rate:
                parents[i].breed(parents[i + 1])
                parents[i + 1].breed(parents[i])

        for ind in offspring:
            if random.random() < offspring.mutation_rate:
                ind.mutate()

        return offspring

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.population.__iter__()


class Darwin(object):
    @staticmethod
    def genetic_algorithm(population, generations, plot_result=False):
        rank = [(population, population.rank())]
        for generation in range(generations):
            children = population.generate()
            rank.append((children, children.rank()))
        return rank

    # @staticmethod
    # def select_best():
