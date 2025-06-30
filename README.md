# SmokeBERT
*SmokeBERT** is a fine-tuned BERT model designed to extract quantitative information from unstructured clinical notes. \
The current version focuses on identifying smoking-related entities: “packs per day”, “cigarettes per day”, “years smoked”, “pack years”, “years since quitting”, and “quit in the year”.\
The code will be released soon.

## Features
- Based on HuggingFace Transformers
- Trained on Medical token classification task
- Model weights saved in `model.safetensors` format

## Datasets


## Prerequisites

- Obtain API keys for HuggingFace.


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/elena145/smokebert.git
cd smokebert
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install transformers torch safetensors
```

## Inference

### 1. Load Model and Tokenizer 

```bash
from transformers import BertTokenizerFast, BertForTokenClassification
import torch

model_path = "shopee-ner" 

tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)
model.eval()
```

### 2. Run Prediction
```bash
def ner_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    label_ids = predictions[0].tolist()

    results = []
    for token, label_id in zip(tokens, label_ids):
        if token not in tokenizer.all_special_tokens:
            results.append((token, model.config.id2label[label_id]))
    return results

# Example
clinical_note = "Patient smoked 2 packs per day for 10 years but quit in 2015."
print(ner_predict(clinical_note))

```

### 4. Optional: Run the provided notebook:
```bash
jupyter notebook   __.ipynb
```


License
