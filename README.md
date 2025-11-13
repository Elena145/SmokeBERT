# SmokeBERT
**SmokeBERT** is a fine-tuned BERT model designed to extract quantitative information from unstructured clinical notes. \

## Features
- Based on HuggingFace Transformers
- Trained on Medical token classification task
- Model weights saved in `model.safetensors` format

## Datasets Used

Our model was trained on a combined corpus of:
- The **Smoking Status Dataset**  (i2b2 challenge and more)
- The [**MIMIC-III Clinical Notes** ](https://physionet.org/content/mimiciii/1.4/)

Manually annotated variables:
- `packs per day`
- `cigarettes per day`
- `years smoked`
- `pack years`
- `years since quitting` (YSQ)
- `quit in the year`


## Prerequisites

- Python 3.8+
- Optional: `.env` file with Hugging Face token for secure downloads


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/elena145/smokebert.git
cd smokebert
```

### 2. Install Dependencies

```bash
pip install transformers torch safetensors
```

## Inference (Sentence)

Example content:

```python
sentence = "QUIT SMOKING TWO YEARS AGO."
```

run `test_sentence.py` and pass the input.
```bash
python test_sentence.py "QUIT SMOKING TWO YEARS AGO."
```

Expected output: 
```yaml
Text: QUIT SMOKING TWO YEARS AGO.
Output: YSQ: TWO YEARS
```

## Optional: Inference (Manually)

### 1. Load Model and Tokenizer 

Download the weights from the [GitHub release page](https://github.com/Elena145/SmokeBERT/releases), and unzip them into a folder named `SavedModel/`.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("./SavedModel")
tokenizer = AutoTokenizer.from_pretrained("./SavedModel")
```

### 2. Run Prediction
```python
sentence = "QUIT SMOKING TWO YEARS AGO."

inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
logits = outputs.logits
pred_ids = logits.argmax(dim=-1)

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
labels = pred_ids[0].tolist()

# Use included formatter
from format_utils import format_predictions, simplified_label_map, paired_labels
print(format_predictions(tokens, labels, simplified_label_map, paired_labels))
```

Expected output: 
```yaml
YSQ: TWO YEARS
```

### 3. Finetuning Cross Validation Code
Script: train_bert.py

License: This project is licensed under the MIT License.
