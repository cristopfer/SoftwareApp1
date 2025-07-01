import numpy as np

class NeuralNetwork:
    def __init__(self, n_inputs, hidden_layers, n_outputs, learning_rate):  
        self.n_inputs = n_inputs
        self.hidden_layers = hidden_layers
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        if not isinstance(hidden_layers, list) or not all(isinstance(x, int) for x in hidden_layers):
            raise ValueError("hidden_layers debe ser una lista de enteros (ej: [2], [3, 2])")

        layer_sizes = [n_inputs] + hidden_layers + [n_outputs]
        for i in range(len(layer_sizes) - 1):
            rng = np.random.default_rng()  
            self.weights.append(rng.standard_normal(size=(layer_sizes[i], layer_sizes[i+1])))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.layer_outputs = [X]
        current_output = X
        
        for i in range(len(self.weights)):
            current_output = self.sigmoid(np.dot(current_output, self.weights[i]) + self.biases[i])
            self.layer_outputs.append(current_output)
        
        return current_output
    
    def train(self, X, y, epochs):
        for _ in range(epochs):
            output = self.forward(X)
            error = y - output
            deltas = [error * self.sigmoid_derivative(output)]
            
            for i in reversed(range(len(self.hidden_layers))):
                delta = deltas[-1].dot(self.weights[i+1].T) * self.sigmoid_derivative(self.layer_outputs[i+1])
                deltas.append(delta)
            
            deltas.reverse()
            
            for i in range(len(self.weights)):
                self.weights[i] += self.layer_outputs[i].T.dot(deltas[i]) * self.learning_rate
                self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * self.learning_rate

def entrenar_red_neuronal(n_inputs, hidden_layers, n_outputs, learning_rate, X, y, epochs):
    if X.shape[1] != n_inputs:
        raise ValueError(f"X debe tener {n_inputs} columnas (features)")
    if y.shape[1] != n_outputs:
        raise ValueError(f"y debe tener {n_outputs} columnas (salidas)")
    
    nn = NeuralNetwork(
        n_inputs=n_inputs,
        hidden_layers=hidden_layers,
        n_outputs=n_outputs,
        learning_rate=learning_rate
    )
    nn.train(X, y, epochs)

    resultados = []
    for i in range(len(X)):
        prediccion = nn.forward(X[i:i+1])[0][0]
        resultados.append({
            "entrada": X[i].tolist(),
            "prediccion": float(prediccion),
            "esperado": int(y[i][0])
        })
    
    return resultados