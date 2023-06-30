import time
from abc import ABC

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from mealpy import Problem
from mealpy.human_based.ICA import OriginalICA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import keras.backend


# Building the neural network
def create_custom_model(input_dim, output_dim, nodes, n=1, name='model'):
    model = Sequential(name=name)
    for i in range(n):
        model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# Building the PSO function and optimization
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


def main():
    nn_list = []
    optimizer_list = []
    for i in range(0, 100):
        keras.backend.clear_session()
        """
        ==========================================
        """
        # Carrega o conjunto de dados Iris
        iris = load_iris()
        x = iris['data']
        y = iris['target']
        enc = OneHotEncoder()
        y = enc.fit_transform(y[:, np.newaxis]).toarray()
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        
        # divide 15% para testes
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

        # divide 15% para validação
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1765)

        n_features = x.shape[1]
        n_classes = y.shape[1]
        """
        ==========================================
        """
        # Constrói a rede neural utilizando a API Keras do TensorFlow
        n_layers = 1
        model = create_custom_model(n_features, n_classes,
                                    10, n_layers)
        model.summary()

        start_time = time.time()
        print('Model name:', model.name)
        history_callback = model.fit(x_train, y_train,
                                     batch_size=5,
                                     epochs=25,
                                     validation_data=(x_test, y_test)
                                     )
        score_nn = model.evaluate(x_test, y_test)
        time_nn = time.time() - start_time
        print('Test loss:', score_nn[0])
        print('Test accuracy:', score_nn[1])
        print("--- %s seconds ---" % time_nn)

        """
        ==========================================
        """

        start_time = time.time()
        shape = get_shape(model)

        # Treina a rede neural utilizando o ICA para otimizar os pesos
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
                          pro_shape=shape, X_train=x_train, Y_train=y_train, minmax="min")

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
        score = model.evaluate(x_val, y_val)

        print('Test loss NN:', score_nn[0])
        print('Test accuracy NN:', score_nn[1])
        print("--- %s seconds NN ---" % time_nn)
        print('Test loss ICA:', score[0])
        print('Test accuracy ICA:', score[1])
        print("--- %s seconds ICA ---" % (time.time() - start_time))

        nn_list.append(score_nn[1])
        optimizer_list.append(score[1])

    print('Custo mínimo: ', min(nn_list))
    print('Custo máximo: ', max(nn_list))
    print('Custo médio: ', np.mean(nn_list))
    print('Custo padrão: ', np.std(nn_list))

    print('Custo otimizado mínima: ', min(optimizer_list))
    print('Custo otimizado máxima: ', max(optimizer_list))
    print('Custo otimizado média: ', np.mean(optimizer_list))
    print('Custo otimizado padrão: ', np.std(optimizer_list))


if __name__ == '__main__':
    main()
