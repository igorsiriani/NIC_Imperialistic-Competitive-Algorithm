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

# Data set transformation

iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.5, random_state=2)
n_features = X.shape[1]
n_classes = Y.shape[1]


## Building the neural network


def create_custom_model(input_dim, output_dim, nodes, n=1, name='model'):
    model = Sequential(name=name)
    for i in range(n):
        model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


print('n_features')
print(n_features)
print('n_classes')
print(n_classes)

n_layers = 1
model = create_custom_model(n_features, n_classes,
                            10, n_layers)
model.summary()

start_time = time.time()
print('Model name:', model.name)
history_callback = model.fit(X_train, Y_train,
                             batch_size=5,
                             epochs=30,
                             validation_data=(X_test, Y_test)
                             )
score_nn = model.evaluate(X_test, Y_test)
time_nn = time.time() - start_time
print('Test loss:', score_nn[0])
print('Test accuracy:', score_nn[1])
print("--- %s seconds ---" % time_nn)


# %%
## Building the PSO function and optimization
def get_shape(model):
    weights_layer = model.get_weights()
    shapes = []
    for weights in weights_layer:
        shapes.append(weights.shape)
    return shapes


def set_shape(weights, shapes):
    new_weights = []
    index = 0
    for shape in shapes:
        if (len(shape) > 1):
            n_nodes = np.prod(shape) + index
        else:
            n_nodes = shape[0] + index
        tmp = np.array(weights[index:n_nodes]).reshape(shape)
        new_weights.append(tmp)
        index = n_nodes
    return new_weights


# %%
start_time = time.time()
shape = get_shape(model)


class ICA(Problem):
    def __init__(self, lb, ub, minmax, name="ICA", pro_model=None, pro_shape=None, X_train=None, Y_train=None, **kwargs):
        super().__init__(lb, ub, minmax, **kwargs)
        self.Y_train = Y_train
        self.X_train = X_train
        self.pro_shape = pro_shape
        self.pro_model = pro_model
        self.name = name

    def fit_func(self, solution):
        try:
            # print('WEIGHTS')
            # result = []
            # for weights in solution:
            self.pro_model.set_weights(self.set_new_shape(solution))
            score = self.pro_model.evaluate(self.X_train, self.Y_train, verbose=0)
            # result.append(1 - score[1])
            return 1 - score[1]

        except AttributeError as error:
            print(error)
            return 50

    def set_new_shape(self, weights):
        new_weights = []
        index = 0
        for shape in self.pro_shape:
            if (len(shape) > 1):
                n_nodes = np.prod(shape) + index
            else:
                n_nodes = shape[0] + index
            tmp = np.array(weights[index:n_nodes]).reshape(shape)
            new_weights.append(tmp)
            index = n_nodes
        return new_weights



# x_max = 1.0 * np.ones(83)
# x_min = -1.0 * x_max
# bounds = (x_min, x_max)
# options = {'c1': 0.3, 'c2': 0.8, 'w': 0.4}
# optimizer = GlobalBestPSO(n_particles=40, dimensions=83,
#                           options=options, bounds=bounds)
# cost, pos = optimizer.optimize(evaluate_nn, 10, X_test=X_train, Y_test=Y_train, shape=shape)

epoch = 100
pop_size = 40
empire_count = 5
assimilation_coeff = 1.5
revolution_prob = 0.05
revolution_rate = 0.1
revolution_step_size = 0.1
zeta = 0.1
x_max = 1.0 * np.ones(83)
x_min = -1.0 * x_max
problem_ica = ICA(lb=x_min, ub=x_max, name="ICA", pro_model=model,
                  pro_shape=shape, X_train=X_train, Y_train=Y_train, minmax="min")

print('problem_ica:')
print(problem_ica)

ica_model = OriginalICA(epoch, pop_size, empire_count, assimilation_coeff, revolution_prob, revolution_rate,
                        revolution_step_size, zeta)
best_position, best_fitness = ica_model.solve(problem=problem_ica)

print('best_fitness: ')
print(best_fitness)
print('best_position: ')
print(best_position)

model.set_weights(set_shape(best_position, shape))
score = model.evaluate(X_test, Y_test)

print('Test loss NN:', score_nn[0])
print('Test accuracy NN:', score_nn[1])
print("--- %s seconds NN ---" % time_nn)
print('Test loss ICA:', score[0])
print('Test accuracy ICA:', score[1])
print("--- %s seconds ICA ---" % (time.time() - start_time))
# # %%
# # Use sphere function
#
# m = Mesher(func=fx.sphere)
# pos_history = [pos[:, :2] for pos in optimizer.pos_history]
# pos3d = m.compute_history_3d(pos_history)
# # Assuming we already had an optimizer ready
# my_animator = Animator(repeat=False)
# my_designer = Designer(figsize=(6, 6))
# animation = plot_surface(pos3d, animator=my_animator, designer=my_designer)
# # %%
# animation.save('pso.gif', writer='imagemagick', fps=6, )
# Image(url='pso.gif')
# # %%
