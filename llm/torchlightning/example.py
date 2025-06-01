

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.demos import WikiText2
from torch.utils.data import DataLoader
from model.net import LitAutoEncoder, Encoder, Decoder, LightningTransformer
from torch.utils.tensorboard import SummaryWriter
def train():
    # dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    # train_loader = DataLoader(dataset)# model
    # autoencoder = LitAutoEncoder(Encoder(), Decoder())

    # # train model
    # trainer = L.Trainer()
    # trainer.fit(model=autoencoder, train_dataloaders=train_loader)


    dataset = WikiText2()
    dataloader = DataLoader(dataset)
    model = LightningTransformer(vocab_size=dataset.vocab_size)

    trainer = L.Trainer(fast_dev_run=10,
                        check_val_every_n_epoch=5, # 训练5次校验一次
                        val_check_interval=0.2  #
                        )
    trainer.fit(model=model, train_dataloaders=dataloader)

def test():
    model_path = 'lightning_logs/version_0/checkpoints/epoch=59-step=3600000.ckpt'



def main():
    train()
    test()


if __name__ == '__main__':
    main()