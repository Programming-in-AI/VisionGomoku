# # Loads the data from the data folder

# import os
# from PIL import Image
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision
# from torchvision import transforms, utils

# # Define the dataset class
# class CUB200(Dataset):
#     def __init__(self, root_dir, training=True, transform=None):
#         self.root_dir = root_dir
#         self.training = training
#         self.transform = transform
#         if training:
#             data_augmentation = transforms.Compose([transforms.Resize((600, 600), transforms.InterpolationMode.BILINEAR),
#                                                    transforms.RandomCrop((448, 448)),
#                                                    transforms.RandomHorizontalFlip(),
#                                                    transforms.ToTensor(),
#                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#         else:
#             data_augmentation = transforms.Compose([transforms.Resize((600, 600), transforms.InterpolationMode.BILINEAR),
#                                                     transforms.CenterCrop((448, 448)),
#                                                     transforms.ToTensor(),
#                                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#         self.dataset = torchvision.datasets.ImageFolder(root_dir, transform=data_augmentation)

#         self.train_dataset = []
#         self.val_dataset = []

#         # Get teh train and validation data
#         line = self.read_txt('../CUB_200_2011/train_test_split.txt')
#         self.train_dataset = [self.dataset[i] for i in range(len(line)) if line[i][-1] == '1']
#         self.val_dataset = [self.dataset[i] for i in range(len(line)) if line[i][-1] == '0']

#     def __len__(self):
#         if self.training:
#             return len(self.train_dataset)
#         else:
#             return len(self.val_dataset)

#     def __getitem__(self, idx):
#         if self.training:
#             return self.train_dataset[idx]
#         else:
#             return self.val_dataset[idx]

#     def read_txt(self, root_dir):
#         f = open(root_dir, mode='r')
#         line = f.read().split('\n')
#         del line[-1]
#         line = sorted(line, key=lambda x: int(x.split(' ')[0]))
#         return line

# # Define the transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # Create the dataset
# dataset = CUB200(root_dir='../CUB_200_2011/images', training=True, transform=transform)

# # Create the dataloader
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)


from torchvision import transforms
import torchvision
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, root_dir, isTrain):
        if isTrain:
            # data_augmentation = transforms.Compose([transforms.Resize((600, 600), InterpolationMode.BILINEAR),
            #                                        transforms.RandomCrop((448, 448)),
            #                                        transforms.RandomHorizontalFlip(),
            #                                        transforms.ToTensor(),
            #                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        else:
            # data_augmentation = transforms.Compose([transforms.Resize((600, 600), InterpolationMode.BILINEAR),
            #                                         transforms.CenterCrop((448, 448)),
            #                                         transforms.ToTensor(),
            #                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])

        dataset = torchvision.datasets.ImageFolder(root_dir, transform=transform)

        self.train_dataset = []
        self.val_dataset = []

        line = read_txt('../CUB_200_2011/CUB_200_2011/train_test_split.txt')
        print('Number of lines: {}'.format(len(line)))
        print('Length of dataset: {}'.format(len(dataset)))
        print('[Training data]')
        self.train_dataset = [dataset[i] for i in tqdm(range(len(line))) if line[i][-1] == '1']

        print()

        print('[Validating data]')
        self.val_dataset = [dataset[i] for i in tqdm(range(len(line))) if line[i][-1] == '0']

    def __len__(self):
        return len(self.train_dataset), len(self.val_dataset)

    def __getitem__(self, idx):
        return self.train_dataset[idx], self.val_dataset[idx]


def read_txt(root_dir):
    f = open(root_dir, mode='r')
    line = f.read().split('\n')
    del line[-1]
    line = sorted(line, key=lambda x: int(x.split(' ')[0]))
    return line


def Dataloader(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader

