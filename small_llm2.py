import torch
import torch.nn as nn
import numpy as np
# Sample text data (you can replace this with your own dataset)
text = "hello world. this is a simple language model example."
# Create a set of unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Unique characters: {vocab_size}")
# Create mappings from characters to integers and vice versa
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}
# Hyperparameters
embedding_dim = 10
hidden_dim = 128
num_layers = 1
sequence_length = 5
num_epochs = 1000
learning_rate = 0.01
# Prepare the dataset
def prepare_data(text, seq_length):
    inputs = []
    targets = []
    for i in range(len(text) - seq_length):
        input_seq = text[i:i + seq_length]
        target_char = text[i + seq_length]
        inputs.append([char_to_idx[char] for char in input_seq])
        targets.append(char_to_idx[target_char])
    return np.array(inputs), np.array(targets)
X, y = prepare_data(text, sequence_length)
# Convert to PyTorch tensors
X_tensor = torch.LongTensor(X)
y_tensor = torch.LongTensor(y)
# Define the RNN model
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Get the output of the last time step
        return out
# Initialize the model, loss function, and optimizer
model = CharRNN(vocab_size, embedding_dim, hidden_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
# Function to generate text
def generate_text(model, start_str, gen_length=50):
    model.eval()
    chars = [char_to_idx[c] for c in start_str]
    input_tensor = torch.LongTensor(chars).unsqueeze(0)  # shape: (1, seq_length)
    generated = start_str
    for _ in range(gen_length):
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, dim=1)
        
        next_char = idx_to_char[predicted_idx.item()]
        generated += next_char
        
        # Prepare the input for the next character prediction
        predicted_idx = predicted_idx.unsqueeze(0)  # shape: (1, 1)
        input_tensor = torch.cat((input_tensor[:, 1:], predicted_idx), dim=1)
    return generated
# Example usage
if __name__ == "__main__":
    start_string = "hello"
    generated_text = generate_text(model, start_string, gen_length=50)
    print(generated_text)