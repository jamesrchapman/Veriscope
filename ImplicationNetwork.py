import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Define the premise and hypothesis statements
premise = "The cat is sitting on the mat."
hypothesis = "The cat is on the mat."

# Tokenize and encode the premise and hypothesis
encoded_inputs = tokenizer.encode_plus(premise, hypothesis, return_tensors='pt', padding=True, truncation=True)

# Forward pass through the model
with torch.no_grad():
    logits = model(**encoded_inputs)[0]

# Compute the degree of implication
entailment_score = torch.softmax(logits, dim=1)[0][1].item()

# Print the degree of implication
print("Entailment Score:", entailment_score)
