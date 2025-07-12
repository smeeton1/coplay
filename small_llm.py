import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Load pre-trained model and tokenizer
model_name = 'gpt2'  # You can use 'gpt2-medium', 'gpt2-large', etc. for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
# Function to generate text
def generate_text(prompt, max_length=50):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    # Generate text
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
# Example usage
if __name__ == "__main__":
    prompt = "Once upon a time"
    generated = generate_text(prompt, max_length=100)
    print(generated)