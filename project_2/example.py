import itertools
import random

import numpy as np
from matplotlib import pyplot as plt

from project_2.genetic_algorithm import Individual, Darwin, Population


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return list(zip(a, b))


class City:
    def __init__(self, name, x, y):
        super().__init__()
        self.name = name
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def clone(self):
        return City(str(self.name), int(self.x), int(self.y))

    def __repr__(self):
        return self.name + "(" + str(self.x) + "," + str(self.y) + ")"


class Path(Individual):

    def __init__(self, chromossome_mutation_rate, *cities):
        super().__init__(chromossome_mutation_rate)
        self.path = list(cities)
        self.fit = 0

    def fitness(self):
        if self.fit == 0:
            for ind_a, ind_b in pairwise(self.path):
                if ind_b:
                    self.fit += ind_a.distance(ind_b)
        return 1 / self.fit if self.fit != 0 else 0.

    def clone(self):
        super().clone()
        r = Path(self.chromossome_mutation_rate, *[city.clone() for city in self.path])
        r.fit = 0
        return r

    def breed(self, other):
        gene_a = int(random.random() * len(self.path))
        gene_b = int(random.random() * len(self.path))

        start_gene = min(gene_a, gene_b)
        end_gene = max(gene_a, gene_b)
        child_path = [ind.clone() for ind in self.path]
        child_path[start_gene:end_gene] = [ind.clone() for ind in other.path[start_gene:end_gene]]
        child = Path(self.chromossome_mutation_rate, *child_path)
        return child

    def mutate(self):
        new_path = self.path.copy()
        for swapped in range(len(self.path)):
            if random.random() < self.chromossome_mutation_rate:
                swap_with = int(random.random() * len(self.path))

                city1 = self.path[swapped].clone()
                city2 = self.path[swap_with].clone()

                new_path[swapped] = city2
                new_path[swap_with] = city1
        return Path(self.chromossome_mutation_rate, *new_path)

    def __repr__(self):
        return ";".join([str(c) for c in self.path])

    @staticmethod
    def generate_random(chromossome_mutation_rate, *cities):
        route = random.sample(cities, len(cities))
        return Path(chromossome_mutation_rate, *route)


class TravellingSalesman(Population):

    def __init__(self, source, size=None, mutation_rate=0., crossover_rate=0.):
        super().__init__(source, size, mutation_rate, crossover_rate)

    def select_breeding_pool(self):
        return self.population[:int(self.size / 2)]

    def rank(self):
        return super().rank()

    def clone(self):
        super().clone()
        return TravellingSalesman([path.clone() for path in self.population], self.size, self.mutation_rate, self.crossover_rate)

    def __repr__(self):
        return ";".join([str(c) for c in self.population])

    @staticmethod
    def generate_random(size, chromossome_mutation_rate, mutation_rate, crossover_rate, *cities):
        population = [Path.generate_random(chromossome_mutation_rate, *cities) for _ in range(size)]
        return TravellingSalesman(population, size=len(population), mutation_rate=mutation_rate, crossover_rate=crossover_rate)


if __name__ == '__main__':
    city_list = []
    for i in range(0, 25):
        city_list.append(City(str(i), x=int(random.random() * 200), y=int(random.random() * 200)))
    print(city_list)
    population = TravellingSalesman.generate_random(100, 0.1, 0.01, 0.2, *city_list)
    result = Darwin.genetic_algorithm(population, 1000)
    sorted_result = sorted(result, key=lambda a: a.fit, reverse=True)

    plot_data = list(map(lambda c: c.fit, result))
    plt.plot(plot_data)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

    plot_data = list(map(lambda c: c.fit, sorted_result))
    plt.plot(plot_data)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

