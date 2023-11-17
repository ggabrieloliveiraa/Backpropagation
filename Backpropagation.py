import numpy as np

# Definindo a função de ativação e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Função para gerar dados de treinamento para as funções lógicas
def generate_data(function, n_inputs):
    # Gerar todas as combinações possíveis de entradas booleanas
    inputs = np.array(np.meshgrid(*[[0, 1]] * n_inputs)).T.reshape(-1, n_inputs)

    # Calcular a saída correspondente para a função lógica escolhida
    if function == 'AND':
        outputs = np.all(inputs, axis=1).astype(int)
    elif function == 'OR':
        outputs = np.any(inputs, axis=1).astype(int)
    elif function == 'XOR':
        outputs = np.bitwise_xor.reduce(inputs, axis=1)
    else:
        raise ValueError("Função desconhecida")

    return inputs, outputs.reshape(-1, 1)

# Definindo a rede neural
class NeuralNetwork:
    def __init__(self, n_inputs):
        # Inicializando pesos aleatoriamente
        self.weights1 = np.random.rand(n_inputs, 4)  # Pesos da camada oculta
        self.weights2 = np.random.rand(4, 1)        # Pesos da camada de saída

    def feedforward(self, inputs):
        self.layer1 = sigmoid(np.dot(inputs, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

    def backpropagation(self, inputs, outputs, learning_rate):
        # Calculando o erro
        error = outputs - self.feedforward(inputs)
        
        # Calculando o delta (ajuste) para os pesos
        d_weights2 = np.dot(self.layer1.T, (2 * error * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(inputs.T, (np.dot(2 * error * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # Atualizando os pesos
        self.weights1 += learning_rate * d_weights1
        self.weights2 += learning_rate * d_weights2

    def train(self, inputs, outputs, learning_rate=0.1, epochs=10000):
        for _ in range(epochs):
            self.backpropagation(inputs, outputs, learning_rate)

# Função para criar e treinar a rede neural
def train_network(function, n_inputs):
    inputs, outputs = generate_data(function, n_inputs)
    neural_network = NeuralNetwork(n_inputs)
    neural_network.train(inputs, outputs)
    return neural_network
class ImprovedNeuralNetwork:
    def __init__(self, n_inputs):
        # Aumentando o número de neurônios nas camadas ocultas
        self.weights1 = np.random.rand(n_inputs, 8)  # Primeira camada oculta
        self.weights2 = np.random.rand(8, 8)         # Segunda camada oculta
        self.weights3 = np.random.rand(8, 1)         # Camada de saída

    def feedforward(self, inputs):
        self.layer1 = sigmoid(np.dot(inputs, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = sigmoid(np.dot(self.layer2, self.weights3))
        return self.output

    def backpropagation(self, inputs, outputs, learning_rate):
        # Calculando o erro
        error = outputs - self.feedforward(inputs)
        
        # Calculando o delta (ajuste) para os pesos
        d_weights3 = np.dot(self.layer2.T, (2 * error * sigmoid_derivative(self.output)))
        d_weights2 = np.dot(self.layer1.T, (np.dot(2 * error * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2)))
        d_weights1 = np.dot(inputs.T, (np.dot(np.dot(2 * error * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # Atualizando os pesos
        self.weights1 += learning_rate * d_weights1
        self.weights2 += learning_rate * d_weights2
        self.weights3 += learning_rate * d_weights3

    def train(self, inputs, outputs, learning_rate=0.1, epochs=50000):
        for _ in range(epochs):
            self.backpropagation(inputs, outputs, learning_rate)

# Criando uma função modificada para treinar a rede neural
def train_improved_network(function, n_inputs):
    inputs, outputs = generate_data(function, n_inputs)
    neural_network = ImprovedNeuralNetwork(n_inputs)
    neural_network.train(inputs, outputs, learning_rate=0.1, epochs=50000)
    return neural_network

# Função para gerar todas as possíveis "n" entradas booleanas
def generate_boolean_inputs(n):
    # Gerar todas as combinações possíveis de entradas booleanas
    return np.array(np.meshgrid(*[[0, 1]] * n)).T.reshape(-1, n)

# Testando a função generate_boolean_inputs com 3 entradas
boolean_inputs_3 = generate_boolean_inputs(3)
boolean_inputs_3

# Exemplo de uso: Treinando uma rede para a função XOR com 3 entradas
# Treinando e testando a rede neural melhorada para a função XOR com 3 entradas
improved_nn_xor = train_improved_network('XOR', 3)
xor_results = improved_nn_xor.feedforward(boolean_inputs_3)
# Treinando redes neurais para as funções AND e OR com 3 entradas
nn_and = train_network('AND', 3)
nn_or = train_network('OR', 3)

# Testando as redes com as entradas geradas
and_results = nn_and.feedforward(boolean_inputs_3)
or_results = nn_or.feedforward(boolean_inputs_3)
print("======================ENTRADAS======================\n")
print(boolean_inputs_3)
print("======================AND======================\n")
print(and_results)
print("======================OR======================\n")
print(or_results)
print("======================XOR======================\n")
print(xor_results)
