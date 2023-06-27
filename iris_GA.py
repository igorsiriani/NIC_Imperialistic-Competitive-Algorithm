import time

import numpy
import numpy as np
import tensorflow
from keras.layers import Dense
from keras.models import Sequential
import pygad
import pygad.kerasga as kerasga
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def fitness_func(ga_instance, solution, sol_idx):
    global x_test, y_test, keras_ga, model
    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)
    score = model.evaluate(x_test, y_test)
    solution_fitness = score[1]

    return solution_fitness


def main():
    global x_test, y_test, keras_ga, model

    nn_list = []
    optimizer_list = []
    for i in range(0, 100):
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
        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled, y, test_size=0.5, random_state=2)
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

        # Criação do modelo genético
        num_generations = 10000
        num_parents_mating = 1000
        keras_ga = kerasga.KerasGA(model=model, num_solutions=num_parents_mating)
        initial_population = keras_ga.population_weights

        # Execução do algoritmo genético
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_func,
                               initial_population=initial_population,
                               crossover_probability=1,
                               mutation_probability=0.1)

        # Execução do algoritmo genético para otimizar os pesos da rede neural
        best_weights, solution_fitness, solution_idx = ga_instance.best_solution()

        # Avaliação do melhor indivíduo
        model.set_weights(set_shape(best_weights, shape))
        score = model.evaluate(x_test, y_test)

        print('Test loss NN:', score_nn[0])
        print('Test accuracy NN:', score_nn[1])
        print("--- %s seconds NN ---" % time_nn)
        print('Test loss GA:', score[0])
        print('Test accuracy GA:', score[1])
        print("--- %s seconds GA ---" % (time.time() - start_time))

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
