import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.efficientnet import get_efficientnet_model
from utils.data_loader import get_datasets

# Load configuration
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Configuration parameters
batch_size = config['train']['batch_size']
num_epochs = config['train']['num_epochs']
learning_rate = config['train']['learning_rate']

# Load data
train_dataset, test_dataset = get_datasets()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, optimizer
model = get_efficientnet_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy: {100 * correct / total}%")
