import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "输出的维度必须能被head整除"
        self.head_num = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.V = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.L1 = nn.Linear(d_out, d_out)
        self.drop = nn.Dropout(dropout)
        # self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length), diagonal=1))
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, t, c = x.shape
        # (b, t, c)
        q = self.Q(x)  
        k = self.K(x)  
        v = self.V(x) 
        # b, t, c -> b, t, h, d
        q = q.view(b, t, self.head_num, self.head_dim)
        k = k.view(b, t, self.head_num, self.head_dim)
        v = v.view(b, t, self.head_num, self.head_dim)
        # b, t, h, d -> b, h, t, d
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        # b, h, t,t
        score = q @ k.transpose(2, 3)
        mask = self.mask.bool()[:t, :t]
        score = score.masked_fill_(mask, -torch.inf)
        weight = torch.softmax(score / c**0.5, dim=-1)
        weight = self.drop(weight)
        # b, h, t, d
        out = weight @ v
        # b,t,h,d
        out = out.transpose(1, 2)
        out = out.contiguous().view(b, t, self.d_out)
        out = self.L1(out)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class TransformerBlock(nn.Module):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.atten = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            num_heads=cfg['n_heads'],
            context_length=cfg['context_length'],
            dropout=cfg['drop_rate'],
        )
        self.ff = FeedFoward(cfg['emb_dim'], cfg['drop_rate'])
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.atten(x)   
        x = self.drop(x)
        x = x + shortcut 
        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x)
        x = x + shortcut  
        return x
        


class BigramLanguageModel(nn.Module):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.L1 = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        __, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.block(x)
        x = self.norm1(x)
        logits = self.L1(x)
        return logits


if __name__ == '__main__':
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }
    print(BigramLanguageModel(cfg=GPT_CONFIG_124M))