import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification
import evaluate
import numpy as np
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import KFold
import re
import collections

nltk.download('punkt_tab')

checkpoint = "medicalai/ClinicalBERT"
# 'bert-base-uncased'
# "dmis-lab/biobert-v1.1"
# "Charangan/MedBERT"
# "medicalai/ClinicalBERT"

# 'cahya/xlm-roberta-base-indonesian-NER'
# 'pritamdeka/BioBert-PubMed200kRCT'
# "dslim/bert-base-NER"


tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load metric
metric = evaluate.load("seqeval")

label_list = ['O', "B-PACKS_PER_DAY","I-PACKS_PER_DAY","B-CIGS_PER_DAY","I-CIGS_PER_DAY",
              "B-YEARS_SMOKED","I-YEARS_SMOKED","B-PACK_YEARS","I-PACK_YEARS",
              "B-YSQ","I-YSQ","B-QUIT_AT_YEAR","I-QUIT_AT_YEAR"]
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {v: k for k, v in id2label.items()}

# Function to label tokens
def label_tokens(row):
    text = str(row['Text']).strip().lower()  # Ensure text is a string
    tokens = tokenizer.tokenize(text)  
    labels = ['O'] * len(tokens)  # Initialize all labels to 'O'

    for col in ['CIGS_PER_DAY', 'PACKS_PER_DAY', 'PACK_YEARS', 'QUIT_AT_YEAR', 'YEARS_SMOKED', 'YSQ']:
    #for col in ["YEARS_SMOKED"]:
        value = row[col]
        entity_str = str(value).strip().lower() if row[col] else ""
        if entity_str:
            entity_list = entity_str.split("; ")
            i = 0 
            while i < len(tokens) and entity_list:
                entity = entity_list[0]  
                entity_tokens = tokenizer.tokenize(entity)  
                entity_len = len(entity_tokens)

                if (
                    i + entity_len <= len(tokens) and
                    ''.join(tokens[i:i + entity_len]).replace("##", "") == ''.join(entity_tokens).replace("##", "")
                ):
                    labels[i] = f"B-{col}"
                    for j in range(1, entity_len):
                        labels[i + j] = f"I-{col}"

                    entity_list.pop(0)
                    # avoid relabeling
                    i += entity_len  
                else: # If no match, move to the next token
                    i += 1  
        # assume that annotated entity that comes first in sentence will also come first in combined_entities.
    
    return {
        'tokens': tokens,
        'Custom_Labels': labels,
        'ner_tags': [label2id[label] for label in labels]
    }

# Function to align labels with tokens
def align_labels_with_tokens(true_labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else true_labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = true_labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer( 
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_list[l] for l in label if l != -100 and l != 0] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100 and l != 0]
        for prediction, label in zip(predictions, labels)
    ]

    accuracy = accuracy_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    report = classification_report(true_labels, true_predictions)

    return {
        "accuracy": accuracy, 
        "f1": f1, 
        "report": report}



# Load the CSV file
df = pd.read_csv('file_example.csv')
df = df.iloc[:-1]
df_new = df.applymap(lambda x: str(x).strip() if pd.notnull(x) else x)

# add a space after "x" --> "x "

if "Text" in df.columns:
    df_new["Text"] = df_new["Text"].astype(str).apply(lambda x: re.sub(r'(\bx)(\d+)', r'\1 \2', x))


# split test data first
raw_test_df = df_new.sample(frac=0.1, random_state=123)
df_new = df_new.drop(raw_test_df.index)

print(f"Length of df_new: {len(df_new)}")

kf = KFold(n_splits=5, shuffle=False, random_state=None)


overall_y_pred = []
overall_y_true = []
all_evaluation_results = []

for train_index, valid_index in kf.split(df_new):
    
    raw_train_df = df_new.iloc[train_index].reset_index(drop=True)
    raw_valid_df = df_new.iloc[valid_index].reset_index(drop=True)

    raw_train = Dataset.from_pandas(raw_train_df)
    raw_valid = Dataset.from_pandas(raw_valid_df)


    raw_train = raw_train.map(label_tokens)
    raw_valid = raw_valid.map(label_tokens)


    tokenized_train = raw_train.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_train.column_names
    )

    tokenized_valid = raw_valid.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_valid.column_names
    )


    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # set to True to use custom labels
    )

    args = TrainingArguments(
        "shopee-ner",
        eval_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=10,  
        weight_decay=0.01,
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    #evaluation_results = trainer.evaluate(tokenized_valid)
    #all_evaluation_results.append(evaluation_results)
    prediction_output = trainer.predict(tokenized_valid)
    logits = prediction_output.predictions
    labels = prediction_output.label_ids

    # Convert logits to predictions
    predictions = np.argmax(logits, axis=-1)

    # Process and align predictions and labels
    fold_labels = []
    fold_predictions = []

    for pred_seq, label_seq in zip(predictions, labels):
        # Remove ignored indices (-100)
        filtered_preds = [
            id2label[pred] for pred, label in zip(pred_seq, label_seq) if label != -100
        ]
        filtered_labels = [
            id2label[label] for label in label_seq if label != -100
        ]

        # Add to fold-specific lists
        fold_predictions.append(filtered_preds)
        fold_labels.append(filtered_labels)
    
    fold_report = classification_report(fold_labels, fold_predictions, output_dict=True)
    all_evaluation_results.append(fold_report)

    # Extend overall lists with fold results
    overall_y_true.extend(fold_labels)
    overall_y_pred.extend(fold_predictions)

# Micro Average
#print(all_evaluation_results)
recall = recall_score(overall_y_true,  overall_y_pred, average="micro")
print("Overall Micro Average Recall: ", recall)
cr = classification_report(overall_y_true, overall_y_pred)
print("Overall Classification report: ", cr)
print("---------------------------------------------------------")

metric_scores = collections.defaultdict(lambda: {'recall': [], 'precision': [], 'f1-score': []})

for report in all_evaluation_results:
    for label in ['PACKS_PER_DAY', 'CIGS_PER_DAY', 'PACK_YEARS', 
                  'QUIT_AT_YEAR', 'YEARS_SMOKED', 'YSQ', 'micro avg']:
        if label in report:
            metric_scores[label]['precision'].append(report[label]['precision'])
            metric_scores[label]['recall'].append(report[label]['recall'])
            metric_scores[label]['f1-score'].append(report[label]['f1-score'])

print("---------------------------------------------------------")
# Print results
for label, metrics in metric_scores.items():
    print(f"\n--- {label} ---")
    for metric_name, scores in metrics.items():
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"{metric_name.capitalize()} - Std: {std:.2f}")


print("---------------------------------------------------------")


# Specify a subdirectory within Folder1
custom_directory = "./SavedModel"

import os
os.makedirs(custom_directory, exist_ok=True)

# Save the model and tokenizer
trainer.save_model(custom_directory)
tokenizer.save_pretrained(custom_directory)

print(f"Model and tokenizer saved to {custom_directory}")

