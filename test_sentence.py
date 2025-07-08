import torch
import pandas as pd

'''
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

torch.manual_seed(42)  # Replace 42 with any number of your choice

# If you're using GPUs, you should also set the seed for the GPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
'''
from format_utils import format_predictions, simplified_label_map, paired_labels

from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load the saved model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("./SavedModel")
tokenizer = AutoTokenizer.from_pretrained("./SavedModel")


sentence = "QUIT SMOKING TWO YEARS AGO."


# Tokenize the sentence
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)  
logits = outputs.logits
# get the predicted class for each token
pred_ids = torch.argmax(logits, dim=-1) 

# Convert token IDs to tokens and numeric labels
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
labels = [p.item() for p in pred_ids[0]]  # numeric labels
formatted_output = format_predictions(tokens, labels, simplified_label_map, paired_labels)

print(f"Text: {sentence}")
print(f"Output: {formatted_output}")
