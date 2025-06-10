# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2025/06/10 15:12:07
@Author  :   orange 
@Version :   1.0
@Contact :   chenorange2219@gmail.com
'''

# here put the import lib


import os
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path
import tiktoken
import torch
from torch.utils.data import DataLoader
torch.manual_seed(123)
from utils.load_smss_dataset import SMSSDataset, SpamDataset
from utils.gpt import GPTModel

def download_and_zip_data(url, zip_path, extracted_path, data_file_path, save_data_file_path):
    if save_data_file_path.exists():
        print("Data already downloaded")
        return
    
    with urllib.request.urlopen(url) as response:
        with open(zip_path, 'wb') as f:
            f.write(response.read())

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"Data downloaded and saved to {data_file_path}")


def read_data(file_path):
    return pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'])

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


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

@torch.no_grad()  # Disable gradient tracking for efficiency
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

def main():
    data_set_url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "ssm_spam_collection.zip"
    extracted_path = "ssm_spam_collection"
    save_data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    download_and_zip_data(data_set_url, zip_path, extracted_path, save_data_file_path, save_data_file_path)
    df = read_data(save_data_file_path)
    balance_df = create_balance_dataset(df)
    map_dict = {
        "ham": 0,
        "spam": 1
    }
    balance_df['label'] = balance_df['label'].map(map_dict)
    train_df, val_df, test_df = random_split(balance_df)

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = SpamDataset(train_df, tokenizer)
    print(train_dataset.max_length)
    eval_dataset = SpamDataset(val_df, tokenizer, max_length=train_dataset.max_length)
    test_dataset = SpamDataset(test_df, tokenizer, max_length=train_dataset.max_length)
    num_workers = 0
    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=False)
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }
    model = GPTModel(GPT_CONFIG_124M)
    # model.load_state_dict("test/weightmodel_10.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # NEW
    model = model.to(device) # NEW
    # print(model)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    epochs = 10
    train_model_simple(    model, train_dataloader, eval_dataloader, optimizer, device,
    num_epochs=epochs, eval_freq=50, eval_iter=5, )
    model._save_to_state_dict

    file_name =  "./weight/" +  f"model_{epochs}.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Saved {file_name}")

if __name__ == "__main__":
    main()