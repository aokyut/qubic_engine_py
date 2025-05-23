import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from time import sleep

class NNEvaluator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x
    
    def training_step(self, batch, batch_idx):
        # sleep(0.1)
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.layer(x)
        loss = F.binary_cross_entropy(z, y)
        loss_base = F.binary_cross_entropy(y, y)
        # print(z, y)
        self.log('train_loss', loss - loss_base)
        return loss - loss_base
    
    def validation_step(self, batch, batch_idx):
        # sleep(0.1)
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.layer(x)
        loss = F.binary_cross_entropy(z, y)
        loss_base = F.binary_cross_entropy(y, y)
        self.log('val_loss', loss - loss_base)
        return loss - loss_base

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class BottoleNeck3d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        neck = ch // 4
        self.layer = nn.Sequential(
            nn.Conv3d(ch, neck, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(neck, neck, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv3d(neck, ch, kernel_size=1)
        )
    
    def forward(self, x):
        y = self.layer(x) + x
        return F.gelu(y)

class DiscEvaluator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 3),
            nn.Softmax(-1)
        )
        self.accuracy = Accuracy(task="multiclass", num_classes=3)

    def forward(self, x):
        # x = x.reshape(-1, 2, 4, 4, 4)
        x = self.layer(x)
        return torch.matmul(x, torch.tensor([0, 0.5, 1]))
    
    def training_step(self, batch, batch_idx):
        sleep(0.1)
        x, y = batch
        # x = x.reshape(-1, 2, 4, 4, 4)
        z = self.layer(x)
        loss = F.cross_entropy(z, y)
        loss_base = F.cross_entropy(y, y)
        # print(z, y)
        self.log('train_loss/entropy', loss - loss_base)
        label = torch.max(y, dim=1).indices
        self.log('train_loss/accuracy', self.accuracy(z, label))
        return loss - loss_base
    
    def validation_step(self, batch, batch_idx):
        # TODO
        sleep(0.1)
        x, y = batch
        # x = x.reshape(-1, 2, 4, 4, 4)
        z = self.layer(x)
        loss = F.cross_entropy(z, y)
        loss_base = F.cross_entropy(y, y)
        self.log('val_loss/entropy', loss - loss_base)
        label = torch.max(y, dim=-1).indices
        self.log('val_loss/accuracy', self.accuracy(z, label))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
