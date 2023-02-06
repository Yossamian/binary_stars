import torch
from datasets import BestDataset_v2
from torch.utils.data import random_split, DataLoader
from trainer import Trainer
import torch.nn as nn
import torch.nn.functional as F

def main():

  full_data = BestDataset_v2()
  train_len = int(len(full_data)*0.8)
  val_len = len(full_data) - train_len

  train_data, val_data = random_split(full_data, [train_len, val_len], generator=torch.Generator().manual_seed(42))

  print('Dataset created:', len(train_data), len(val_data))

  train_loader = DataLoader(train_data, batch_size=1)
  val_loader = DataLoader(val_data, batch_size=1)

  model = ConvolutionalNet(n_params=12)
  epochs = 10
  optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.001)

  trainer = Trainer(
    model=model,
    epochs=epochs,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader
  )

  trainer.train()

# model definition
class ConvolutionalNet(nn.Module):

    # define model elements
    def __init__(self, n_params):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 16, 5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(16, 32, 5, stride=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 5, stride=5, padding=2)
        self.conv4= nn.Conv1d(64, 64, 5, stride=5, padding=2)
        self.conv_final= nn.Conv1d(64, 5, 1, stride=1, padding=0)

        # Linear layers
        self.linear1 = nn.Linear(410,100)
        self.linear2 = nn.Linear(100, n_params)
        
    # forward propagate input
    def forward(self, X):

        # unsqueeze to fit dimension for nn.Conv1d
        X = torch.unsqueeze(X, 1)

        # Four convolutional layers
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = F.relu(self.conv4(X))

        # 1D convolution, then flatten for FC layers
        X = F.relu(self.conv_final(X))
        X = torch.flatten(X, start_dim=1)

        # Two FC layers to give output
        X = F.relu(self.linear1(X))
        out = self.linear2(X)

        return out


if __name__=="__main__":
  main()