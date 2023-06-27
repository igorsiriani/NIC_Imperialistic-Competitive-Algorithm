# %%
import time
from abc import ABC

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from mealpy import Problem
from pyswarms.single.global_best import GlobalBestPSO
from mealpy.human_based.ICA import OriginalICA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pygmo as pg


class ICA(Problem):
    def __init__(self, lb, ub, minmax, name="ICA", **kwargs):
        super().__init__(lb, ub, minmax, **kwargs)
        self.name = name

    def fit_func(self, solution):
        x = solution[0]
        y = solution[1]
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


epoch = 100
pop_size = 50
empire_count = 5
assimilation_coeff = 1.5
revolution_prob = 0.05
revolution_rate = 0.1
revolution_step_size = 0.1
zeta = 0.1
x_max = 5.0 * np.ones(2)
x_min = -5.0 * x_max
problem_ica = ICA(lb=x_min, ub=x_max, name="ICA", minmax="min")

print('problem_ica:')
print(problem_ica)

ica_model = OriginalICA(epoch, pop_size, empire_count, assimilation_coeff, revolution_prob, revolution_rate,
                        revolution_step_size, zeta)
best_position, best_fitness = ica_model.solve(problem=problem_ica)

print('best_fitness: ')
print(best_fitness)
print('best_position: ')
print(best_position)
