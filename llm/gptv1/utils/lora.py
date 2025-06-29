import torch
import torch.nn as nn


class LoRALinear(nn.Module):

    def __init__(self, linear_layer, in_channel, out_channel,rank=8, alpha=32,) -> None:
        super().__init__()
        self.base_layer = linear_layer
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.rank = rank
        self.alpha = alpha

        # freazen
        self.base_layer.weight.requires_grad = False
        # if hasattr(se)

        self.lora_A = nn.Parameter(torch.zeros((rank, self.in_channel)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_channel, rank)))

        nn.init.normal_(self.lora_A, std=1/rank)
        nn.init.normal_(self.lora_B)

    def forward(self, x):
        output = (self.lora_B @ self.lora_A @ x.T).T * (self.alpha / self.rank)
        return self.base_layer(x) + output