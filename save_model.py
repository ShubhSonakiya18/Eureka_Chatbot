import torch
import nltk
from nltk.tokenize import word_tokenize
import torch.nn as nn
import math

# Load the trained model
model = torch.load('gpt2_finetuned.pth', map_location=torch.device('cpu'))
model.eval()

# Define the vocabulary
word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}

# Add words from your training data
conversations = [
    ("Hi!", "Hello! How can I help you?"),
    ("What is your name?", "I am a chatbot created from scratch."),
    ("How are you?", "I'm doing well, thank you!"),
    ("Bye", "Goodbye! Have a great day!")
]

def tokenize(sentence):
    return word_tokenize(sentence.lower())

# Build vocabulary
for input_text, target_text in conversations:
    for word in tokenize(input_text) + tokenize(target_text):
        if word not in word2idx:
            index = len(word2idx)
            word2idx[word] = index
            idx2word[index] = word

vocab_size = len(word2idx)

# Save the model and vocabulary together
checkpoint = {
    'model_state_dict': model.state_dict(),
    'word2idx': word2idx,
    'idx2word': idx2word,
    'vocab_size': vocab_size
}

torch.save(checkpoint, 'transformer_chatbot1.pth')
print("Model and vocabulary saved successfully!") 