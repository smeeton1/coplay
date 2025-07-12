import jax
import jax.numpy as jnp
from jax import grad, jit, random
# Generate synthetic data
def generate_data(num_samples=100):
    x = jnp.linspace(-10, 10, num_samples)
    y = 2 * x + 1 + jax.random.normal(random.PRNGKey(0), (num_samples,))  # y = 2x + 1 + noise
    return x, y
# Define the model
def model(params, x):
    return params[0] * x + params[1]  # y = mx + b
# Define the loss function (Mean Squared Error)
def loss(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)
# Training function
@jit
def train(x, y, params, learning_rate=0.01, num_epochs=1000):
    for _ in range(num_epochs):
        grads = grad(loss)(params, x, y)
        params = params - learning_rate * grads
    return params
# Main function to run the training
def main():
    x, y = generate_data()
    
    # Initialize parameters (slope and intercept)
    params = jnp.array([0.0, 0.0])  # Initial guess for [m, b]
    
    # Train the model
    trained_params = train(x, y, params)
    
    print("Trained parameters (slope, intercept):", trained_params)
if __name__ == "__main__":
    main()