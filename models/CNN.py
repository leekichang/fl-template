import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_class=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Linear(32*32*32, n_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
if __name__ == '__main__':
    model = CNN(n_class=10)
    x = torch.randn((2,3,32,32))
    o = model(x)
    print(o.shape)
    