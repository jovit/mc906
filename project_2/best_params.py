import itertools
import os
import random

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from project_2.genetic_algorithm import Individual, Darwin, Population
from project_2.read_data import read_data


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return list(zip(a, b))


class Model(Individual):
    # [layers(0-4), neurons(100-2000), numfeatures(200, 1000), loss(binary_crossentropy, mse)]
    def __init__(self, chromossome_mutation_rate, input_size, layers, neurons, num_features, train_set, test_set):
        super().__init__(chromossome_mutation_rate)
        self.input_size = input_size
        self.layers = layers
        self.neurons = neurons
        self.num_features = num_features
        self.train_set = train_set
        self.test_set = test_set
        self.loss = 0
        self.model = None

    def fitness(self):
        if not self.model:
            layers = [tf.keras.layers.Dense(self.input_size)]
            for i in range(self.layers // 2):
                layers.append(tf.keras.layers.Dense(self.neurons[i], activation=tf.nn.relu))
            layers.append(tf.keras.layers.Dense(self.num_features, activation=tf.nn.sigmoid))
            for i in range(self.layers // 2, len(self.neurons)):
                layers.append(tf.keras.layers.Dense(self.neurons[i], activation=tf.nn.relu))
            layers.append(tf.keras.layers.Dense(self.input_size, activation=tf.nn.sigmoid))

            self.model = tf.keras.models.Sequential(layers)

            self.model.compile(optimizer='adadelta', loss='binary_crossentropy')

            self.model.fit(self.train_set, self.train_set, epochs=2, validation_data=(self.test_set, self.test_set))
            self.loss = self.model.evaluate(self.test_set, self.test_set)

        return 1. / self.loss

    def clone(self):
        super().clone()
        return Model(self.chromossome_mutation_rate, self.input_size, self.layers, np.copy(self.neurons), self.num_features, self.train_set, self.test_set)

    def breed(self, other):
        def __layers_crossover(child):
            child.layers = other.layers
            neurons_len = len(child.neurons)
            if neurons_len < child.layers:
                child.neurons = np.concatenate((child.neurons, np.copy(other.neurons[neurons_len - child.neurons:])))
            elif neurons_len > child.layers:
                child.neurons = child.neurons[:child.layers]

        def __neurons_crossover(child):
            neurons_len = min(len(self.neurons), len(other.neurons))
            idx_a = random.randint(0, neurons_len - 1)
            idx_b = random.randint(0, neurons_len - 1)

            start = min(idx_a, idx_b)
            end = max(idx_a, idx_b)
            child_neurons = np.copy(self.neurons)
            child_neurons[start:end] = np.copy(other.neurons[start:end])
            child.neurons = child_neurons

        def __num_features_crossover(child):
            child.num_features = other.num_features

        child = self.clone()
        crossover_functions = [__layers_crossover, __neurons_crossover, __num_features_crossover]
        crossover_features = random.randint(1, len(crossover_functions))
        for func in random.choices(crossover_functions, crossover_features):
            func(child)
        return child

    def mutate(self):
        def __mutate_layers(other):
            other.layers = np.random.randint(0, 4)
            neurons_len = len(other.neurons)
            if neurons_len < other.layers:
                other.neurons = np.concatenate((other.neurons, np.random.random_integers(100, 2000, size=other.layers - neurons_len)))
            elif neurons_len > other.layers:
                other.neurons = other.neurons[:other.layers]

        def __mutate_neurons(other):
            for n in range(len(other.neurons)):
                if random.random() < self.chromossome_mutation_rate:
                    other.neurons[n] = np.random.randint(100, 2000)

        def __mutate_num_features(other):
            other.num_features = np.random.randint(200, 1000)

        other = self.clone()
        mutate_functions = [__mutate_layers, __mutate_neurons, __mutate_num_features]
        mutate_features = random.randint(1, len(mutate_functions))
        for func in random.choices(mutate_functions, mutate_features):
            func(other)
        return other

    def __repr__(self):
        return "Layers: {}, Neurons: {}, # Features: {}, Loss: {}".format(self.layers, self.neurons, self.num_features, self.loss)

    @staticmethod
    def generate_random(chromossome_mutation_rate, train_set, test_set):
        layers = np.random.randint(0, 4)
        neurons = np.random.random_integers(100, 2000, layers)
        num_features = np.random.randint(200, 1000)
        return Model(chromossome_mutation_rate, 50 * 50, layers, neurons, num_features, train_set, test_set)


class FaceAutoencoderModels(Population):

    def __init__(self, source, size=None, mutation_rate=0., crossover_rate=0.):
        super().__init__(source, size, mutation_rate, crossover_rate)

    def select_breeding_pool(self):
        return self.population[:int(self.size / 2)]

    def rank(self):
        return super().rank()

    def clone(self):
        super().clone()
        return FaceAutoencoderModels([model.clone() for model in self.population], self.size, self.mutation_rate, self.crossover_rate)

    def __repr__(self):
        return ";".join([str(c) for c in self.population])

    @staticmethod
    def generate_random(size, chromossome_mutation_rate, mutation_rate, crossover_rate, train_set, test_set):
        population = [Model.generate_random(chromossome_mutation_rate, train_set, test_set) for _ in range(size)]
        return FaceAutoencoderModels(population, size=len(population), mutation_rate=mutation_rate, crossover_rate=crossover_rate)


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    tf.random.set_random_seed(1)

    train = read_data('imgs/train_aligned/').astype(float)
    train = train / 255.

    test = read_data('imgs/test/').astype(float)
    test = test / 255.

    train = np.array([it.flatten() for it in train])
    test = np.array([it.flatten() for it in test])

    population = FaceAutoencoderModels.generate_random(5, 0.1, 0.2, 0.6, train, test)
    result = Darwin.genetic_algorithm(population, 100)

    plot_data = list(map(lambda c: c.loss, result))
    plt.plot(plot_data)
    plt.ylabel('Loss')
    plt.xlabel('Generation')
    plt.show()
