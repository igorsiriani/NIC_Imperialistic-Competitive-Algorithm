import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import load_iris
import keras.backend
from keras.layers import Dense
from keras.models import Sequential
from pyswarms.single.global_best import GlobalBestPSO


# Função que altera o shape
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


# Função que busca o shape de cada layer
def get_shape(model):
    weights_layer = model.get_weights()
    shapes = []
    for weights in weights_layer:
        shapes.append(weights.shape)
    return shapes


# Função de aptidão
def evaluate_nn(W, shape, X_test, Y_test, model):
    print('WEIGHTS')
    print(W.shape)
    print(W)

    result = []
    for weights in W:
        model.set_weights(set_shape(weights, shape))
        score = model.evaluate(X_test, Y_test, verbose=0)
        result.append(1 - score[1])
    return result


# Função de criar modelo personalizado
def create_custom_model(input_dim, output_dim, nodes, n=1, name='model'):
    model = Sequential(name=name)
    for i in range(n):
        model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


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

        # Treina a rede neural utilizando o PSO para otimizar os pesos
        start_time = time.time()
        shape = get_shape(model)
        x_max = 1.0 * np.ones(83)
        x_min = -1.0 * x_max
        bounds = (x_min, x_max)
        options = {'c1': 0.3, 'c2': 0.8, 'w': 0.4}
        optimizer = GlobalBestPSO(n_particles=40, dimensions=83, options=options, bounds=bounds)
        cost, pos = optimizer.optimize(evaluate_nn, 100, X_test=x_train, Y_test=y_train, shape=shape, model=model)
        model.set_weights(set_shape(pos, shape))
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
