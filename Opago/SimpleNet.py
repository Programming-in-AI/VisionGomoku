import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 맨처음에 3개가 들어오잖아 RGB, 32개필터를 만들어버려, 필터사이즈는 3이야
        # self.conv1 = nn.Conv2d(input, output, filtersize, stride=2, padding=1)
        self.conv1 = nn.Conv2d(1, 32, 7, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 7, padding='same')
        self.conv3 = nn.Conv2d(64, 128, 7, padding='same')
        self.conv4 = nn.Conv2d(128, 64, 7, padding='same')
        self.conv5 = nn.Conv2d(64, 32, 7, padding='same')
        self.conv6 = nn.Conv2d(32, 1, 7, padding='same')
        self.sg = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, x):
        # x = x  # batch_size x 1 x 15 x 15

        x = self.conv1(x)  # batch_size x 32 x 15 x 15
        x = self.relu(x)

        x = self.conv2(x)  # batch_size x 64 x 15 x 15
        x = self.relu(x)

        x = self.conv3(x)  # bach_size x 128 x 15 x 15
        x = self.relu(x)

        x = self.conv4(x)  # batch_size x 64 x 15 x 15
        x = self.relu(x)

        x = self.conv5(x)  # batch_size x 32 x 15 x 15
        x = self.relu(x)

        x = self.conv6(x)  # batch_size x 1 x 15 x 15

        x = x.view(-1, 15 * 15)  # batch_size x 225
        x = self.sg(x)  # batch_size x 225, which one is very high?

        return x