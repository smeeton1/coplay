import numpy as np
# Generate synthetic data
def generate_synthetic_data(num_samples=60000, num_classes=10, img_size=28*28):
    # Randomly generate features
    X = np.random.rand(num_samples, img_size)
    # Randomly generate labels
    y = np.random.randint(0, num_classes, size=num_samples)
    # One-hot encode labels
    y_one_hot = np.eye(num_classes)[y]
    return X, y_one_hot, y
# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)
# Neural Network class
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Weights initialization
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
    def forward(self, x):
        self.hidden_layer = sigmoid(np.dot(x, self.weights_input_hidden))
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output))
        return self.output_layer
    def backward(self, x, y, learning_rate):
        output_error = y - self.output_layer
        output_delta = output_error * sigmoid_derivative(self.output_layer)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer)
        # Update weights
        self.weights_hidden_output += self.hidden_layer.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += x.T.dot(hidden_delta) * learning_rate
    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(x)
            self.backward(x, y, learning_rate)
# Generate synthetic dataset
train_images, train_labels_one_hot, train_labels = generate_synthetic_data(num_samples=60000)
test_images, test_labels_one_hot, test_labels = generate_synthetic_data(num_samples=10000)
# Initialize and train the neural network
input_size = 28 * 28
hidden_size = 128
output_size = 10
nn = SimpleNeuralNetwork(input_size, hidden_size, output_size)
nn.train(train_images, train_labels_one_hot, epochs=10, learning_rate=0.1)
# Evaluate the model
predictions = nn.forward(test_images)
predicted_labels = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_labels == test_labels)
print(f'Test accuracy: {accuracy * 100:.2f}%')