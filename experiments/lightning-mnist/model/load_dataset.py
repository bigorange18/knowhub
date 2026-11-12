import os
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

class OfficialDataset(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
        train_dataset, val_dataset, test_dataset = random_split(dataset, [50000, 5000, 5000])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        ...

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=True)



class PersonalDataset(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        ...
        self.train_dataset = None

    def setup(self):
        ...

    def __len__(self):
        return len(self.train_dataset)
    

    def __getitem__(self, idx):
        return self.train_dataset[idx]