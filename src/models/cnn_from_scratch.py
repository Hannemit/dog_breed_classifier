import torch.nn as nn
import torch.nn.functional as F

num_classes = 133  # the number of different outcome classes in our current problem


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # layers of a CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)

        # fully connected layers
        self.fc1 = nn.Linear(256 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, num_classes)

        self.dropout = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """
        Define forward behaviour.
        :param x: image input, here assumed to be size 3x224x224 (i.e. a 224x224 colour image)
        :return: output after a forward pass through the network, an array of length num_classes
        """
        # Define forward behavior
        x = self.conv1_bn(self.pool(F.relu(self.conv1(x))))  # 3x224x224 ---> 16x112x112

        x = self.conv2_bn(self.pool(F.relu(self.conv2(x))))  # 16x112x112 ---> 32x56x56
        # x = F.relu(self.conv2b(x)) #64x56x56 ---> 64x56x56
        x = self.conv3_bn(self.pool(F.relu(self.conv3(x))))  # 32x56x56 ---> 64x28x28
        # x = F.relu(self.conv3b(x)) #128x28x28 ---> 128x28x28
        x = self.conv4_bn(self.pool(F.relu(self.conv4(x))))  # 64x28x28 ---> 128x14x14
        x = self.conv5_bn(self.pool(F.relu(self.conv5(x))))  # 128x14x14 ---> 256x7x7

        # fully connected layers
        x = x.view(-1, 256 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        # x = self.dropout(F.relu(self.fc3(x)))

        x = self.fc3(x)

        return x
