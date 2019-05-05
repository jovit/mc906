import random


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
        Mutates self based on another individual
        :param other: mate partner
        :return: its child
        """
        pass

    def mutate(self):
        """
        Mutates self
        :return: a new individual based on self
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
        self.population.sort(key=lambda x: x.fitness(), reverse=True)

    def select_breeding_pool(self):
        pass

    def select_next_gen(self):
        self.population.sort(key=lambda x: x.fitness(), reverse=True)
        self.population = self.population[:self.size]

    def clone(self):
        pass

    def rank(self):
        return self.population[0].fitness()

    def best_individual(self):
        return self.population[0]

    def generate(self):
        offspring = self.clone()
        parents = offspring.select_breeding_pool()
        for i in range(1, len(parents) - 1, 2):
            if random.random() < offspring.crossover_rate:
                offspring.population.append(parents[i].breed(parents[i + 1]))
                offspring.population.append(parents[i + 1].breed(parents[i]))

        mutated_individuals = []
        for ind in offspring:
            if random.random() < offspring.mutation_rate:
                mutated_individuals.append(ind.mutate())
        offspring.population.extend(mutated_individuals)

        offspring.select_next_gen()
        return offspring

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return self.population.__iter__()


class Darwin(object):
    @staticmethod
    def genetic_algorithm(population, generations, should_end=None):
        best_individuals = [population.best_individual()]
        next_gen = population
        for generation in range(generations):
            if should_end and should_end(best_individuals):
                break
            next_gen = next_gen.generate()
            best_individuals.append(next_gen.best_individual())
        return best_individuals
