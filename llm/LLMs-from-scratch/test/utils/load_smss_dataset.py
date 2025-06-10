# -*- encoding: utf-8 -*-
'''
@File    :   load_smss_dataset.py
@Time    :   2025/06/10 16:20:52
@Author  :   orange 
@Version :   1.0
@Contact :   chenorange2219@gmail.com
'''


import pandas as pd
import tiktoken
import torch
from torch.utils.data import  Dataset

class SMSSDataset(Dataset):
    def __init__(self, csv_file:pd.DataFrame, tokenizer, max_len=None, pad_token_id=50256):
        self.data = csv_file
        self.encode_texts = [tokenizer.encode(text) for text in self.data['text']]
    
        if max_len is None:
            self.max_len = self._longest_encode_length()
        else:
            self.max_len = max_len
            self.encode_texts = [
                self.encode_texts[:self.max_len]
            ]
    
    def __getitem__(self, index):
        encode = self.encode_texts[index]
        label = self.data.iloc[index]['label']
        return torch.tensor(encode, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _longest_encode_length(self):
        max_length = 0
        for encode in self.encode_texts:
            encode_length = len(encode)
            if encode_length > max_length:
                max_length = encode_length
        # return max([len(encode) for encode in self.encode_texts])
        return max_length
    


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        # self.data = pd.read_csv(csv_file)
        self.data = csv_file

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length