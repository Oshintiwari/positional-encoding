import torch
import torch.nn as nn
import torch.nn.functional as F

# Dummy Dataset
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, seq_len=10, input_dim=16):
        self.data = torch.randn(num_samples, seq_len, input_dim)
        self.labels = torch.randint(0, 2, (num_samples,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Learnable Positional Encoding
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.randn(seq_len, d_model))

    def forward(self, x):
        # Add positional embeddings to the input
        return x + self.position_embeddings

# Transformer Model with Learnable Positional Encoding
class TransformerModel(nn.Module):
    def __init__(self, seq_len, input_dim, num_classes, d_model=16, num_heads=4, num_layers=2):
        super(TransformerModel, self).__init__()
        self.positional_encoding = LearnablePositionalEncoding(seq_len, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Aggregate sequence info
        x = self.fc(x)
        return x

# Hyperparameters
seq_len = 10
input_dim = 16
num_classes = 2
batch_size = 16
num_epochs = 5

# Load Dataset
dataset = DummyDataset(num_samples=100, seq_len=seq_len, input_dim=input_dim)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
model = TransformerModel(seq_len=seq_len, input_dim=input_dim, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(num_epochs):
    for data, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training Complete")
