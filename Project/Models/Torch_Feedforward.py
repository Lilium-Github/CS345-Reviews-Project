import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3, num_layers=4):
        super(FeedForward, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class FeedForwardTrainer:
    def __init__(self, hidden_size=64, epochs=10, batch_size=32, lr=0.001, dropout=0.3):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def train(self, features, labels):
        if hasattr(features, "toarray"):
            features = features.toarray()

        # Remap labels to start from 0
        unique = sorted(set(labels))
        label_map = {v: i for i, v in enumerate(unique)}
        labels = np.array([label_map[l] for l in labels])

        print(f"Classes: {unique} → mapped to 0-{len(unique)-1}")
        X = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_size = X.shape[1]
        output_size = len(torch.unique(y))

        self.model = FeedForward(input_size, self.hidden_size, output_size).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}  Loss: {total_loss/len(loader):.4f}")

    def predict(self, features):

        if hasattr(features, "toarray"):
            features = features.toarray()

        X = torch.tensor(features, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            predictions = torch.argmax(outputs, dim=1)
        return predictions.cpu().numpy()