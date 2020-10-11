import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.dense1 = nn.Linear(7*7*64, 256)
        self.dense2 = nn.Linear(256, 10)

    def forward(self, data):
        x = F.relu(self.conv1(data))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, p=0.25, training=self.training)
        flattened = x.view(-1, 7*7*64)
        x = F.relu(self.dense1(flattened))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.dense2(x)
        output = F.log_softmax(x)
        return output

