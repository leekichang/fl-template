import copy
import torch
import utils

import config as cfg

class BaseClient(object):
    def __init__(self, args, name):
        self.args      = args
        self.exp_name  = args.exp_name
        self.name      = name
        self.model     = None # Assigned in Server Class
        self.trainset  = None # Assigned in Server Class
        self.testset   = None # Assigned in Server Class
        self.save_path = f'./checkpoints/{self.exp_name}'
        self.device    = 'cuda' if torch.cuda.is_available() else 'cpu'
        utils.ensure_path(self.save_path)

    def get_name(self):
        return self.name

    def get_type(self):
        return 'Base'
    
    def train(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
    def save_model(self, tag):
        torch.save(self.model.state_dict(), f'{self.save_path}/{self.name}_{tag}.pt')