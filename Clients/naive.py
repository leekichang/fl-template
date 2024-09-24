import torch
import torch.nn as nn


import utils
import Clients
from DataManager import datamanager

class NaiveClient(Clients.BaseClient):
    def __init__(self, args, name):
        super(NaiveClient, self).__init__(args, name)
        self.tag  = 'Client'

    def setup(self):
        self.model       = utils.build_model(self.args)
        self.criterion   = nn.CrossEntropyLoss()
        self.optimizer   = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.lr*0.1)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True , drop_last=True )
        self.testloader  = torch.utils.data.DataLoader(self.testset , batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        self.n_samples   = len(self.trainset)
        self.epoch       = self.args.epoch

    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        for epoch in range(self.epoch):
            for idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()
        self.model = self.model.to('cpu')

    def test(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            correct, total, loss = 0, 0, 0
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                acc = 100*correct/total
                loss += self.criterion(outputs, target).item()
            print(f'{self.tag} {self.name:<3} Accuracy: {acc:.2f}%')
        self.model = self.model.to('cpu')
        return loss/len(self.testloader), acc
    
if __name__ == '__main__':
    pass