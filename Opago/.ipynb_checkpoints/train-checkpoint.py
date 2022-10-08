import os
import torch.nn
from dataloader import Dataloader
from SimpleNet import *
from cnn_utils import *

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: {}".format(device))

# dataload
data_path = os.getcwd()+'/dataset/'
x_train, x_val, y_train, y_val = Dataloader(data_path)


net = SimpleNet()


epoch = 1
learning_rate = 0.0002
loss_function = torch.nn.CrossEntropyLoss().to(device)
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5, 0.99), weight_decay=0.1)
# optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate, weight_decay=0.1)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.1)


net = net.to(device)
loss_list = []

# model summary

train_losses, train_acc, val_acc = train_net(net, trainloader, validloader, optimizer, epoch, device, loss_function)
