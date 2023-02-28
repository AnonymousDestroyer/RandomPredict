# -*- coding: utf-8 -*-
# @Time    : 2023/2/28 2:32
# @Author  : Yujin Wang

import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from torch.utils.data import random_split

cpu_device = 'cpu'
# data = torch.randint(1, 255, [10000, 100]).cpu().numpy()
# labels = torch.randint(1, 255, [10000]).cpu().numpy()


pd_data = pd.read_csv('./data/t800w.csv').to_numpy().squeeze()
data = pd_data

window_size = 101
stride = 3

data_list = []
labels_list = []

for i in range(0, len(data) - window_size + 1, stride):
    data_list.append(data[i:i+window_size-1])
    labels_list.append(data[i+window_size-1])

data = np.array(data_list)[:100000]
labels = np.array(labels_list)[:100000]
print('==> data & labels processed')
print("train_data shape:", data.shape)  # (31, 100)
print("labels shape:", labels.shape)  # (31,)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        x = torch.LongTensor(x).to(device)
        y = torch.LongTensor([y]).to(device)
        return x, y

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_size)
        self.pos_embedding = nn.Embedding(100, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(hidden_size, output_dim)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.hidden_size)
        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        positions = self.pos_embedding(positions).unsqueeze(0).expand(x.size())
        x = x + positions
        x = self.encoder(x)
        x = self.decoder(x[:, 0, :])
        return x


BATCH_SIZE = 256
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

dataset = MyDataset(data, labels)
train_data_size = int(0.8 * (len(dataset)))
test_data_size = len(dataset) - train_data_size

train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_data_size, test_data_size])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = TransformerModel(input_dim=256, output_dim=256).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # data and labels
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    acc = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            acc += accuracy_score(labels.detach().cpu().numpy().flatten(), preds.detach().cpu().numpy().flatten())
    print(f"Epoch {epoch + 1}, loss: {running_loss / len(train_dataset)}, accuracy: {acc / len(test_dataloader)}")

# save model
torch.save(model, 'model.pth')
print('saved')
