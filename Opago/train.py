import os
import torch.nn
from dataloader import *
from SimpleNet import *
from cnn_utils import *
import platform

torch.manual_seed(42)

if platform.system() == 'Darwin':
    device = 'mps'
elif platform.system() == 'Windows':
    device = 'cuda' if torch.cuda.is_available() else  'cpu'

print("Device: {}".format(device))

# data load
data_path = os.getcwd()+'/dataset/'
x_train, x_val, y_train, y_val = Dataloader(data_path)

tr_dataset = BasicDataset(x_train,y_train)
train_loader = data.DataLoader(dataset=tr_dataset, batch_size=128, num_workers = 0, shuffle=True)
val_dataset = BasicDataset(x_val, y_val)
valid_loader = data.DataLoader(dataset=val_dataset, batch_size=128, num_workers = 0, shuffle=True)


# net
net = SimpleNet()
epoch = 30
learning_rate = 0.001
loss_function = torch.nn.BCELoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate )
# optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate, weight_decay=0.1)
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.1)

# train
net = net.to(device)
loss_list = []

# model summary
train_losses, train_acc, val_acc = train_net(net, train_loader, valid_loader, optimizer, epoch, device, loss_function)
