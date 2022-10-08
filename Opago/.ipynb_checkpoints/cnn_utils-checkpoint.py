import tqdm
import os
import torch

def train_net(net, traindata, test_loader, optimizer, epoch, device, loss_fn):
    train_losses = []
    train_acc = []
    val_acc = []

    # model save path
    os.makedirs('./models/', exist_ok=True)

    for epoch in range(epoch):
        running_loss = 0.0
        # train mode
        net.train()

        total = 0
        n_acc = 0

        for i, data in tqdm.tqdm(enumerate(traindata), total=len(traindata)):  # data 안에는 batchsize 만큼의 data 가 들어있고(ex 32개) 그 32개가 net안에 들어가는 거다! 그리고 정확도는 그 32개를 평균치

            net = net.to(device)
            img = img.to(device)
            label = label.to(device)

            h = net(data)

            loss = loss_fn(h, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = data.size(0) # data should be a tensor
            running_loss += loss.item()
            total += batch_size

            _, y_pred = h.max(1)
            n_acc += (label == y_pred).float().sum().item()
        train_losses.append(running_loss / i)

        # train_dataset acc
        train_acc.append(n_acc / total)

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
    ypreds = []
    for x, y in data_loader:
        # send to device
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            _, y_pred = net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)


    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)

    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()




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