import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

file_path = os.path.dirname(__file__)
docs_path = os.path.join(file_path, "../../docs/")
encodings_path = os.path.join(docs_path, "results/encodings/")
model_path = os.path.join(docs_path, "results/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = BertForSequenceClassification.from_pretrained(docs_path)
tokenizer = BertTokenizer.from_pretrained(docs_path)
model.to(device)


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        if prediction:
            return "Not Fake"
        else:   return "Fake"
    return prediction

