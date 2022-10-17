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

        # x already has batch_size of data
        for i, (x, y) in tqdm.tqdm(enumerate(traindata), total=len(traindata)):  # x 안에는 batchsize 만큼의 data 가 들어있고(ex 32개) 그 32개가 net안에 들어가는 거다! 그리고 정확도는 그 32개를 평균치

            net = net.to(device)
            data = x.to(device)
            label = y.to(device)  # label.size = (batch_size, 1, 15, 15)

            h = net(data)  # h.size = (batch_size, 1, 225) throughout a sigmoid function
            loss = loss_fn(h, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = data.size(0)  # data should be a tensor
            running_loss += loss.item()
            total += batch_size

            n_acc += ( torch.argmax(label, dim=1) == torch.argmax(h, dim=1) ).sum().item()
        train_losses.append(running_loss / i)

        # train_dataset acc
        train_acc.append(n_acc / total)  # metric = %

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
            y_pred = net(x)  # net(x).size = (batch_size, 255)
        batch_size = x.size(0)
        total += batch_size
        n_acc += (torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).sum().item() # same => 1 not same => 0

    acc = n_acc/total

    return acc

