#  This file contains the code for training and evaluating the BirdViT model on the CUB-200-2011 dataset.

from dataloader import CustomDataset, Dataloader

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import utils

# Define the architecture of the model
class BirdViT(nn.Module):
    def __init__(self, num_classes=200, image_size=224, patch_size=16, num_channels=3, dim=1024, depth=6, heads=8, mlp_dim=2048, dropout=0.1, emb_dropout=0.1):
        super(BirdViT, self).__init__()
        num_patches = (image_size // patch_size) ** 2 # (224 // 16) ** 2 = 14 ** 2 = 196
        patch_dim = num_channels * patch_size ** 2 # 3 * 16 ** 2 = 768
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # self.transformer = nn.Transformer(dim=dim, depth=depth, num_heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        self.transformer = nn.Transformer(d_model=int(dim),
                                          nhead=int(heads),
                                          num_encoder_layers=int(depth), 
                                          num_decoder_layers=int(depth), 
                                          dim_feedforward=int(mlp_dim), 
                                          dropout=dropout, 
                                          activation="gelu")
        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        p = self.patch_size # 16
        x = x.unfold(2, p, p).unfold(3, p, p) # (batch_size, num_channels, num_patches, patch_dim) = (batch_size, 3, 14, 16, 16)
        print('After unfold: ', x.shape)
        x = x.contiguous().view(x.size(0), x.size(1), -1, p * p) 
        print('After view: ', x.shape)
        x = x.permute(0, 2, 1, 3).contiguous()
        print('After permute: ', x.shape)
        # x = x.view(x.size(0), -1, x.size(-1))
        x = x.view(x.size(0), x.size(1), -1)
        print('After view: ', x.shape)
        x = self.to_patch_embedding(x)
        print('After to_patch_embedding: ', x.shape)

        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        print('After cls_tokens: ', cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        print('After cat: ', x.shape)
        x += self.pos_embedding[:, :(n + 1)]
        print('After pos_embedding: ', x.shape)
        x = self.dropout(x)
        print('After dropout: ', x.shape)

        x = self.transformer(x, x)
        print('After transformer: ', x.shape)
        x = self.to_cls_token(x[:, 0])
        print('After to_cls_token: ', x.shape)
        return self.mlp_head(x)

# Define the training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    return epoch_loss, epoch_acc

# Define the evaluation function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    return epoch_loss, epoch_acc

# Define the main function
def main():
    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Set the random seed
    torch.manual_seed(0)

    # Set the hyperparameters
    num_classes = 200
    image_size = 224
    patch_size = 16
    num_channels = 3
    dim = 1024
    depth = 6
    heads = 8
    mlp_dim = 2048
    dropout = 0.1
    emb_dropout = 0.1
    batch_size = 32
    num_epochs = 10
    learning_rate = 3e-4
    weight_decay = 1e-4

    # Load the dataset
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # Define the training and testing datasets
    # Load the data
    print('Processing dataset...')
    root_dir = '../CUB_200_2011/CUB_200_2011/images/'
    dataset = CustomDataset(root_dir, isTrain=True)
    print("Training data size : {}".format(dataset.__len__()[0]))
    print("Validating data size : {}".format(dataset.__len__()[1]))

    train_dataset = dataset.train_dataset
    val_dataset = dataset.val_dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    # Create the model
    model = BirdViT(num_classes=num_classes, image_size=image_size, patch_size=patch_size, num_channels=num_channels, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout)
    model = model.to(device)
    print(model)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the model
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        end_time = time.time()
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\tVal. Loss: {val_loss:.3f} | Val. Acc: {val_acc*100:.2f}%")

    # Plot the training and validation curves
    utils.plot(train_losses, val_losses, "Loss", "loss.png")
    utils.plot(train_accs, val_accs, "Accuracy", "accuracy.png")

    # Save the model
    torch.save(model.state_dict(), "bird_vit.pth")

if __name__ == "__main__":
    main()