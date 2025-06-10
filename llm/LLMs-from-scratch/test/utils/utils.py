import torch
import os

def saveModle(model, path):
    torch.save(model.state_dict(), path)