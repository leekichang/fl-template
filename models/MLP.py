import torch
import torch.nn as nn

class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearReLU, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)
    
    def forward(self, x):
        return self.dropout(self.relu(self.fc(x)))

class MLP(nn.Module):
    def __init__(self, n_class=10):
        super(MLP, self).__init__()
        self.fc1 = LinearReLU(784, 256)
        self.fc2 = LinearReLU(256, 256)
        self.fc3 = nn.Linear(256, n_class)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
if __name__ == '__main__':
    model = MLP(n_class=10)
    x = torch.randn((64,1,28,28))
    o = model(x)
    print(o.shape)
    