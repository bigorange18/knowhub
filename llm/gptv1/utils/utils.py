import torch
import os
from pathlib import Path
import urllib
import zipfile
import pandas as pd
import tiktoken
from .load_smss_dataset import SpamDataset
from torch.utils.data import DataLoader
def mkdir_folder(folder_path:Path):
    if folder_path.exists():
        return True
    else:
        return folder_path.mkdir()

def load_online_dataset_zip(url, folder:Path, name:str, file_name:str):
    if folder.is_dir():
        if any(folder.iterdir()):
            for file_path in folder.iterdir():
                if '.csv' in str(file_path):
                    print("Data already downloaded")
                    return file_path
    else:
        mkdir_folder(folder)

    folder_zip = folder / f"{name}.zip"
    with urllib.request.urlopen(url) as response:
        with open(folder_zip, 'wb') as f:
            f.write(response.read())

    with zipfile.ZipFile(folder_zip, 'r') as zip_ref:
        zip_ref.extractall(folder)
    file_path = (folder / file_name).rename(folder / (file_name +".csv"))
    print(f"Data downloaded and saved to {file_path}")
    return file_path


def create_balance_dataset(df:pd.DataFrame):
    spam_num = df[df['label'] == 'spam'].shape[0]
    ham_subset  = df[df['label'] == 'ham'].sample(spam_num, random_state=123)
    # print(ham_subset)
    balanced_df = pd.concat([ham_subset, df[df['label'] == 'spam']])
    return balanced_df

def random_split(df:pd.DataFrame, train_ratio=0.7, val_ration=0.2):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    tran_end = int(len(df) * train_ratio)
    val_end = tran_end + int(len(df) * val_ration)
    train_df = df[:tran_end]
    val_df = df[tran_end:val_end]
    test_df = df[val_end:]
    return train_df, val_df, test_df

def data_load_SMS_dataset(file_path, train_ratio=0.7, val_ration=0.2):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'])
    balance_df = create_balance_dataset(df)
    map_dict = {
        "ham": 0,
        "spam": 1
    }
    balance_df['label'] = balance_df['label'].map(map_dict)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    tran_end = int(len(df) * train_ratio)
    val_end = tran_end + int(len(df) * val_ration)
    train_df = df[:tran_end]
    val_df = df[tran_end:val_end]
    test_df = df[val_end:]


    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = SpamDataset(train_df, tokenizer)
    eval_dataset = SpamDataset(val_df, tokenizer, max_length=train_dataset.max_length)
    test_dataset = SpamDataset(test_df, tokenizer, max_length=train_dataset.max_length)
    num_workers = 0
    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=False)
    return train_dataloader, eval_dataloader, test_dataloader


def saveModle(model, path):
    torch.save(model.state_dict(), path)