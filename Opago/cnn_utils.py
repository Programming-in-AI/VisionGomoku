import tqdm
import os
import torch

def train_net(net, traindata, test_loader, optimizer, epoch, device, loss_fn):
    train_losses = []
    train_acc = []
    val_acc = []

    # model save path
    os.makedirs('./models/', exist_ok=True)
    val_acc.append(eval_net(net, test_loader, device))
    for epoch in range(epoch):
        running_loss = 0.0
        # train mode
        net.train()
        total = 0
        n_acc = 0

        # 이미 x는 batch_size만큼 포함중
        for i, (x, y) in tqdm.tqdm(enumerate(traindata), total=len(traindata)):  # x 안에는 batchsize 만큼의 data 가 들어있고(ex 32개) 그 32개가 net안에 들어가는 거다! 그리고 정확도는 그 32개를 평균치

            net = net.to(device)
            data = x.to(device)
            label = y.to(device)  # label.size = (batch_size, 1, 15, 15)

            h = net(data)  # h.size = (batch_size, 1, 225) throughout a sigmoid =>
            loss = loss_fn(h, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = data.size(0) # data should be a tensor
            running_loss += loss.item()
            total += batch_size

           # _, y_pred = h.max(1)
           #  print('h type', type(h))
           #  print('h size', h.size())
           #  # print('type of y_pred',type(y_pred))
           #  # print('size of y_pred', y_pred.size())
           #  print('size of label', label.size())
           #  # print('y_pred len:', len(y_pred))
           #  print('label len:',len(label))
           #  # print('y_pred[0] len:',len(y_pred[0]))
           #  print('label[0] len:',len(label[0]))
            n_acc += ( torch.argmax(label, dim=1) == torch.argmax(h, dim=1) ).sum().item()  # batch_size 만큼
        train_losses.append(running_loss / i)

        # train_dataset acc
        train_acc.append(n_acc / (total * data.size(2) * data.size(3)))  # 퍼센트

        # valid_dataset acc
        val_acc.append(eval_net(net, test_loader, device))
        # epoch
        print(f'epoch: {epoch+1}, train_loss:{train_losses[-1]}, train_acc:{train_acc[-1]},val_acc: {val_acc[-1]}', flush=True)

        #model save
        if epoch % 3 == 0 and epoch > 10:
            torch.save(net.cpu().state_dict(),'./models/model_'+str(epoch)+'.pth')

    return train_losses, train_acc, val_acc


def eval_net(net, data_loader, device):
    # Dropout or BatchNorm 没了
    net.eval()
    ys = []
    total = 0
    n_acc = 0

    for x, y in data_loader:
        # send to device
        x = x.to(device)  # x.size() = (batch_size, 1, 15, 15)
        y = y.to(device)

        with torch.no_grad():
            y_pred = net(x)  # net(x).size = (batch_size, 1, 15, 15)
        batch_size = x.size(0)
        total += batch_size
        n_acc += (y_pred == y).float().sum().item() # 같으면 1 틀리면 0 다 합쳤을때

    acc = n_acc/ (total * x.size(2) * x.size(3))

    return acc




def val(net, device, current_epoch, validloader, criterion):  # Function to validate the network
    net.eval()
    running_loss = 0.0
    total = 0
    for i, data in enumerate(validloader, 0):
        images, label = data
        grays = rgb_to_grayscale(images)
        images = images.to(device)
        grays = grays.to(device)
        outputs = net(grays)
        loss = criterion(outputs, images)
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    average_loss = running_loss / total
    net.train()

    return average_loss