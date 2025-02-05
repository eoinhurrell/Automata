import logging
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# Import our model definition
from transformers import AutoTokenizer, AutoModel


class SequentialWorkflowClassifier(nn.Module):
    def __init__(self, num_labels=60, max_steps=5):
        super().__init__()
        # Use DistilBERT as base model - good balance of performance and speed
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")

        # Changed to take full sequence representation
        self.step_decoder = nn.LSTM(
            input_size=768,  # BERT hidden size
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Adjusted for bidirectional LSTM
        self.step_projections = nn.ModuleList(
            [nn.Linear(512, num_labels) for _ in range(max_steps)]
        )

        self.max_steps = max_steps

    def forward(self, input_ids, attention_mask):
        # Get full BERT output sequence
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Process through LSTM
        lstm_output, _ = self.step_decoder(bert_output)

        # Global max pooling
        sequence_repr = torch.max(lstm_output, dim=1)[0]

        # Project each step using the pooled representation
        step_outputs = []
        for projection in self.step_projections:
            step_output = projection(sequence_repr)
            step_outputs.append(step_output)

        return torch.stack(step_outputs, dim=1)


class WorkflowDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_steps=5):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_steps = max_steps

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_sequence = self.labels[idx]

        # Pad label sequence to max_steps
        padded_labels = label_sequence + [0] * (self.max_steps - len(label_sequence))

        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


class WorkflowTrainer:
    def __init__(self, num_labels=66, max_steps=5, device="cuda"):
        self.device = device
        self.num_labels = num_labels
        self.max_steps = max_steps

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = SequentialWorkflowClassifier(num_labels, max_steps)
        self.model.to(device)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_data(self, texts, labels):
        """Load and split data into train/val/test"""
        dataset = WorkflowDataset(texts, labels, self.tokenizer, self.max_steps)

        # 70/15/15 split
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        return train_dataset, val_dataset, test_dataset

    def train(
        self, train_dataset, val_dataset, batch_size=32, epochs=10, learning_rate=2e-5
    ):
        """Train the model"""
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        criterion = CrossEntropyLoss()

        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_steps = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in progress_bar:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask)

                # Calculate loss for each step
                loss = 0
                for step in range(self.max_steps):
                    step_loss = criterion(outputs[:, step], labels[:, step])
                    loss += step_loss

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_steps += 1

                progress_bar.set_postfix(
                    {"training_loss": f"{train_loss/train_steps:.3f}"}
                )

            # Validation phase
            val_loss, val_accuracy = self.evaluate(val_loader)

            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss/train_steps:.3f} - "
                f"Val Loss: {val_loss:.3f} - "
                f"Val Accuracy: {val_accuracy:.3f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "./model/model.pt")

    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        criterion = CrossEntropyLoss()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask)

                # Calculate loss and predictions for each step
                batch_loss = 0
                batch_preds = []

                for step in range(self.max_steps):
                    step_loss = criterion(outputs[:, step], labels[:, step])
                    batch_loss += step_loss

                    step_preds = torch.argmax(outputs[:, step], dim=1)
                    batch_preds.append(step_preds.cpu().numpy())

                total_loss += batch_loss.item()
                all_preds.extend(np.array(batch_preds).T)
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(
            np.array(all_labels).flatten(), np.array(all_preds).flatten()
        )

        return total_loss / len(dataloader), accuracy

    def predict(self, text):
        """Predict workflow steps for a single text input"""
        self.model.eval()

        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        # Get predictions for each step
        predictions = []
        for step in range(self.max_steps):
            step_pred = torch.argmax(outputs[:, step], dim=1)
            pred_label = step_pred.item()

            # Stop if we predict padding token (0)
            if pred_label == 0:
                break

            predictions.append(pred_label)

        return predictions


# Example usage
def main():

    workflows = pd.read_csv("./data/workflows_10000.csv")
    orig_steps = pd.read_csv("./data/workflow_steps_10000.csv")

    steps = orig_steps.merge(pd.read_csv("./data/valid_apps.csv"))
    steps = steps.merge(pd.read_csv("./data/valid_actions_with_triggers.csv"))
    steps["step"] = steps["app_name"] + "_" + steps["action_name"]
    step_keys = steps["step"].unique()
    pickle.dump(step_keys, open("./model/step_names.pickle", "wb"))
    num_labels = len(steps["step"].unique())

    # steps = steps.groupby("workflow_id")["step"].apply(list)
    # workflows = workflows.merge(steps.reset_index())
    #
    workflows = workflows.merge(steps)
    workflows["there"] = workflows.apply(
        lambda x: x["workflow_description"]
        .lower()
        .replace(" ", "_")
        .find(x["app_name"])
        != -1,
        axis=1,
    )
    workflows = workflows[workflows["there"] == True]
    workflows = (
        workflows.groupby(["workflow_id", "workflow_description"])["step"]
        .apply(list)
        .reset_index()
    )
    workflows["len"] = workflows["step"].apply(len)
    max_steps = int(workflows["len"].max())
    workflows["label"] = workflows["step"].apply(
        lambda x: [list(step_keys).index(y) for y in x]
    )

    texts = workflows["workflow_description"].tolist()
    labels = workflows["label"].tolist()

    # Initialize trainer
    trainer = WorkflowTrainer(
        num_labels=num_labels,
        max_steps=max_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Load and split data
    train_dataset, val_dataset, test_dataset = trainer.load_data(texts, labels)

    # Train model
    trainer.train(
        train_dataset, val_dataset, batch_size=32, epochs=10, learning_rate=2e-5
    )

    # Example prediction
    sample_text = "When I receive a Gmail message, create a Trello card"
    predictions = trainer.predict(sample_text)
    print(f"Predicted workflow steps: {predictions}")


if __name__ == "__main__":
    main()
