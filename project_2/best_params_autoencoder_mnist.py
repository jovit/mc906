import itertools
import os
import random

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from genetic_algorithm import Individual, Darwin, Population
from read_data import read_data

MIN_NEURONS = 10
MAX_NEURONS = 1000
MIN_LAYERS = 1
MAX_LAYERS = 4
MIN_FEAT = 20
MAX_FEAT = 100
MIN_EPOCHS = 1
MAX_EPOCHS = 10


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return list(zip(a, b))


class Model(Individual):
    # [layers(0-4), neurons(100-2000), numfeatures(200, 1000), loss(binary_crossentropy, mse)]
    def __init__(self, chromossome_mutation_rate, input_size, number_of_epochs, layers, neurons, num_features, train_set, test_set):
        super().__init__(chromossome_mutation_rate)
        self.input_size = input_size
        self.layers = layers
        self.neurons = neurons
        self.num_features = num_features
        self.train_set = train_set
        self.test_set = test_set
        self.loss = 0
        self.accuracy = 0
        self.model = None
        self.number_of_epochs = number_of_epochs

    def fitness(self):
        tf.random.set_random_seed(1)

        if self.loss == 0:
            layers = [tf.keras.layers.Dense(self.input_size)]
            for i in range(self.layers // 2):
                layers.append(tf.keras.layers.Dense(self.neurons[i], activation=tf.nn.relu))
            layers.append(tf.keras.layers.Dense(self.num_features, activation=tf.nn.sigmoid))
            for i in range(self.layers // 2, len(self.neurons)):
                layers.append(tf.keras.layers.Dense(self.neurons[i], activation=tf.nn.relu))
            
            layers.append(tf.keras.layers.Dense(self.input_size, activation=tf.nn.sigmoid))

            self.model = tf.keras.models.Sequential(layers)

            self.model.compile(optimizer='adadelta', loss='binary_crossentropy')

            self.model.fit(self.train_set, self.train_set, epochs=self.number_of_epochs, validation_data=(self.test_set, self.test_set))
            self.loss = self.model.evaluate(self.test_set, self.test_set)

        return 1. / self.loss

    def clone(self):
        super().clone()
        doppelganger = Model(self.chromossome_mutation_rate, self.input_size, self.number_of_epochs, self.layers, np.copy(self.neurons), self.num_features, self.train_set,
                             self.test_set)
        doppelganger.loss = self.loss
        return doppelganger

    def breed(self, other):
        def __layers_crossover(child):
            child.layers = other.layers
            neurons_len = len(child.neurons)
            if neurons_len < child.layers:
                child.neurons = np.concatenate((child.neurons, np.copy(other.neurons[neurons_len - child.layers:])))
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

        def __num_epochs_crossover(child):
            child.number_of_epochs = other.number_of_epochs

        child = self.clone()
        crossover_functions = [__neurons_crossover, __num_features_crossover, __num_epochs_crossover]
        crossover_features = random.randint(1, len(crossover_functions))
        for func in random.choices(crossover_functions, k=crossover_features):
            func(child)
            child.loss = 0
            child.model = None
        return child

    def mutate(self):
        def __mutate_layers(other):
            other.layers = np.random.randint(MIN_LAYERS, MAX_LAYERS)
            neurons_len = len(other.neurons)
            if neurons_len < other.layers:
                other.neurons = np.concatenate((other.neurons, np.random.random_integers(MIN_NEURONS, MAX_NEURONS, size=other.layers - neurons_len)))
            elif neurons_len > other.layers:
                other.neurons = other.neurons[:other.layers]

        def __mutate_neurons(other):
            for n in range(len(other.neurons)):
                if random.random() < self.chromossome_mutation_rate:
                    other.neurons[n] = np.random.randint(MIN_NEURONS, MAX_NEURONS)

        def __mutate_num_features(other):
            other.num_features = np.random.randint(MIN_FEAT, MAX_FEAT)

        def __mutate_num_epochs(other):
            other.num_features = np.random.randint(MIN_EPOCHS, MAX_EPOCHS)

        other = self.clone()
        mutate_functions = [__mutate_layers, __mutate_neurons, __mutate_num_features, __mutate_num_epochs]
        mutate_features = random.randint(1, len(mutate_functions))
        for func in random.choices(mutate_functions, k=mutate_features):
            func(other)
            other.loss = 0
            other.model = None
        return other

    def __repr__(self):
        return "Layers: {}, Neurons: {}, Loss: {}, Epochs: {}".format(self.layers, self.neurons, self.loss, self.number_of_epochs)

    @staticmethod
    def generate_random(chromossome_mutation_rate, train_set, test_set):
        layers = np.random.randint(MIN_LAYERS, MAX_LAYERS)
        neurons = np.random.random_integers(MIN_NEURONS, MAX_NEURONS, layers)
        num_features = np.random.randint(MIN_FEAT, MAX_FEAT)
        num_epochs = np.random.randint(MIN_EPOCHS, MAX_EPOCHS)
        return Model(chromossome_mutation_rate, 28 * 28, num_epochs, layers, neurons, num_features, train_set, test_set)


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


def end_alg(best):
    if len(best) > 15:
        threshold = .0
        last_gens = np.array([i.loss for i in best[-5:-1]])
        calc = np.array([i.loss for i in best[-4:]])
        delta = last_gens - calc
        idx = np.where(delta > threshold)
        return len(idx[0]) == 0
    else:
        return False

def mutate_01(individual):
    return individual.mutate()



def crossover_01(individual, partner):
    return individual.breed(partner)



if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    tf.random.set_random_seed(1)

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # train = read_data('imgs/train_aligned/').astype(float)
    # train = train / 255.

    # test = read_data('imgs/test/').astype(float)
    # test = test / 255.

    x_train = np.array([it.flatten() for it in x_train])
    x_test = np.array([it.flatten() for it in x_test])

    population = FaceAutoencoderModels.generate_random(size=10, chromossome_mutation_rate=0.1, mutation_rate=0.2, crossover_rate=0.6,
                                                       train_set=x_train, test_set=x_test)
    result = Darwin.genetic_algorithm(population, 100, mutation_function=mutate_01, crossover_function=crossover_01)
    print("Best: {}".format(result[-1]))
    plot_data = list(map(lambda c: c.loss, result))
    plt.plot(plot_data)
    plt.ylabel('Loss')
    plt.xlabel('Generation')
    plt.show()
