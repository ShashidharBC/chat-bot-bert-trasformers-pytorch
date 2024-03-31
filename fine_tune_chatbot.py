import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class ConversationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_input = row['user_input']
        response = row['response']
        inputs = self.tokenizer(user_input, response, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        inputs['labels'] = torch.tensor([1])  # Wrap the label in a list
        return inputs



def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Load and preprocess data
    data = pd.read_csv("conversation_data.csv")
    train_data, test_data = train_test_split(data, test_size=0.2)

    # Define custom dataset
    class ConversationDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=128):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            user_input = row['user_input']
            response = row['response']
            inputs = self.tokenizer(user_input, response, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
            inputs['labels'] = torch.tensor(1)  # Classification label, assuming all responses are correct
            return inputs

    # Create dataloaders
    batch_size = 8
    train_dataset = ConversationDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(3):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].squeeze(1)  # Remove the extra dimension
            attention_mask = batch['attention_mask'].squeeze(1)  # Remove the extra dimension
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{3}, Loss: {total_loss:.4f}")

    model.save_pretrained("fine_tuned_chatbot_model")

if __name__ == "__main__":
    main()

