import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Load data

file_path = os.path.dirname(__file__)
docs_path = os.path.join(file_path, "../../docs/")
encodings_path = os.path.join(docs_path, "results/encodings/")
model_path = os.path.join(docs_path, "results/")


raw_data = pd.read_csv(docs_path+"news_ru.csv")
data = raw_data[["title", "is_fake"]]

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(data["title"], data["is_fake"], test_size=0.3, shuffle=True)
train_texts = train_texts.reset_index(drop=True)
test_texts = test_texts.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
test_labels = test_labels.reset_index(drop=True)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Encode data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

# Dataset class
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = NewsDataset(train_encodings, train_labels)
test_dataset = NewsDataset(test_encodings, test_labels)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.to(device)    

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-7, weight_decay=0.01)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# TensorBoard writer
writer = SummaryWriter('runs/news_classification_experiment')

# Training loop
num_epochs = 100


def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    return avg_loss

def evaluate(model, test_loader, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    writer.add_scalar('Loss/eval', avg_loss, epoch)
    writer.add_scalar('Accuracy/eval', accuracy, epoch)
    return avg_loss, accuracy

try:
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, epoch)
        eval_loss, eval_accuracy = evaluate(model, test_loader, epoch)
        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}')
        
        # Perform any necessary operations here
        # For example: Log additional metrics, adjust learning rate, etc.
        print(f"Operations after epoch {epoch + 1}")

except KeyboardInterrupt:
    print("\nTraining interrupted. Saving the model...")

finally:
    # Save the model and tokenizer
    model.save_pretrained(docs_path)
    tokenizer.save_pretrained(docs_path)

    # Close the TensorBoard writer
    writer.close()

    print(f"Model and tokenizer saved to {docs_path}")
