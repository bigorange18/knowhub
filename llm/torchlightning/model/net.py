

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)
    


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

from lightning.pytorch.demos import Transformer
class LightningTransformer(L.LightningDataModule):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)


    def forward(self, inputs, target):
        return self.model(inputs, target)
    
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = F.nll_loss(output, target.view(-1))
        return loss
    
    def training_step_end(self, batch, batch_idx):
        ...


    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("loss:", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)