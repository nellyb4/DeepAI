### 3 CNN MODELS ###
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch.nn.functional as F


# convolutional network with 7x7 kernel
class ConvNet7(nn.Module):
    def __init__(self, num_classes=5):
        super(ConvNet7, self).__init__()
        # Define the convolutional layer
        self.conv_net = nn.Sequential(

                                 nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),

                                 nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),

                                 nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                 nn.Dropout(0.3),

                                 # add layers:
                                #  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding='same'),
                                #  nn.BatchNorm2d(128),
                                #  nn.ReLU(),
                                #  nn.Dropout(0.3),
                                 nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),

                                 nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU(),

                                 nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(8),
                                 nn.ReLU(),

                                 nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                 nn.Dropout(0.3),

                                 )
        self.fc = nn.Sequential(
                                nn.Linear(8 * 7 * 7, 128),
                                nn.ReLU(),

                                nn.Linear(128, 64),
                                nn.ReLU(),

                                nn.Linear(64, num_classes),
                                nn.ReLU(),
                                )




    def forward(self, x):
        # pass the input through the convolutional layer
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x


# deeper convolutional network (more layers)
class ConvNet7_deeper(nn.Module):
    def __init__(self, num_classes=5):
        super(ConvNet7_deeper, self).__init__()  # Corrected this line
        # Define the convolutional layer
        self.conv_net = nn.Sequential(

                                 nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),

                                 nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),

                                 nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                 nn.Dropout(0.3),

                                 # add layers:
                                 nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),

                                 nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU(),

                                 nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                 nn.Dropout(0.3),

                                 nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(7, 7), stride=1, padding='same'),
                                 nn.BatchNorm2d(8),
                                 nn.ReLU(),

                                 nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                 nn.Dropout(0.3),

                                 )
        self.fc = nn.Sequential(
                                nn.Linear(8 * 7 * 7, 128),
                                nn.ReLU(),

                                nn.Linear(128, 64),
                                nn.ReLU(),

                                nn.Linear(64, num_classes),
                                nn.ReLU(),
                                )




    def forward(self, x):
        # Pass the input through the convolutional layer
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x


# model = ConvNet()
# output = model(images)


 # convolutional network with 3x3 kernel (more layers)
class ConvNet3_deeper(nn.Module):
    def __init__(self, in_channels=1, out_channels=5):
        super(ConvNet3_deeper, self).__init__()  # Corrected this line

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 48x48 -> 24x24
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout(p=0.3)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.dropout5 = nn.Dropout(p=0.3)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 24x24 -> 12x12
        self.dropout6 = nn.Dropout(p=0.3)

        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.dropout7 = nn.Dropout(p=0.3)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn8 = nn.BatchNorm2d(16)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 12x12 -> 6x6
        self.dropout8 = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(in_features=6*6*16, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=32)
        self.fc3 = nn.Linear(32, out_channels)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x) # <- block 1
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x) # <- block 2
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.dropout3(x) # <- block 3

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x) # <- block 4
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x) # <- block 5
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.dropout6(x) # <- block 6

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.dropout7(x) # <- block 7
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = self.dropout8(x) # <- block 8

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x



class ConvNet3(nn.Module):
    def __init__(self, in_channels=1, out_channels=5):
        super(ConvNet3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 48x48 -> 24x24
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout(p=0.3)

        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.dropout7 = nn.Dropout(p=0.3)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn8 = nn.BatchNorm2d(16)
        self.max_pool3 = nn.MaxPool2d(kernel_size=4, stride=2) # 12x12 -> 6x6
        self.dropout8 = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(in_features=5*5*16, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=32)
        self.fc3 = nn.Linear(32, out_channels)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x) # <- block 1
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x) # <- block 2
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.dropout3(x) # <- block 3

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x) # <- block 4
        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = F.relu(x)
        # x = self.dropout5(x) # <- block 5
        # x = self.conv6(x)
        # x = self.bn6(x)
        # x = F.relu(x)
        # x = self.max_pool2(x)
        # x = self.dropout6(x) # <- block 6

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.dropout7(x) # <- block 7
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = self.dropout8(x) # <- block 8

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x
