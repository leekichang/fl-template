import copy
import torch
from datetime import datetime
from tqdm import tqdm
import torch.utils.tensorboard as tb

import utils
from models import *
from Clients import *
from DataManager import *
from Servers import aggregator

__all__ = ['BaseServer']

class BaseServer(object):
    def __init__(self, args):
        self.args            = args
        self.exp_name        = utils.format_args(self.args)
        self.clients         = []
        self.global_model    = utils.build_model(self.args)
        self.device          = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.aggregator      = getattr(aggregator, self.args.aggregator)()
        self.n_clients       = self.args.n_clients
        self.global_models   = {-1: self.global_model}
        self.n_clusters      = 0
        self.criterion       = utils.build_criterion(args)
        self.round           = 0
        self.setup()
        
    def setup(self):
        print(f"Experiment: {self.exp_name}")
        self.use_tb    = self.args.use_tb
        self.save_path = f'./checkpoints/{self.exp_name}'
        self.prepare_dataset()
        self.init_clients()
        self.dispatch()
        self.TB_WRITER    = tb.SummaryWriter(f'./tensorboard/{str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_{self.exp_name}') if self.use_tb else None
    
    def tb_update(self, round, **kwargs):
        if self.use_tb:
            for key, value in kwargs.items():
                self.TB_WRITER.add_scalar(key, value, round)

    def prepare_dataset(self):
        self.trainset, self.testset = getattr(datamanager, self.args.dataset)()
        self.client_trainsets, self.client_testsets = Dirichlet(self.trainset, 
                                                                self.testset,
                                                                self.n_clients,
                                                                self.args.alpha).split_dataset()
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        
    def init_clients(self):
        print(f"Initializing {self.n_clients} clients")

        for cidx in tqdm(range(self.n_clients)):
            self.clients.append(self.create_client(cidx))
            self.clients[cidx].trainset = copy.deepcopy(self.client_trainsets[cidx])
            self.clients[cidx].testset  = copy.deepcopy(self.client_testsets[cidx])
            self.clients[cidx].setup() 
            
    def create_client(self, client_id):
        return getattr(Clients, self.args.client)(self.args, client_id)
        
    
    def sample_clients(self, n_participants):
        sampled_clients_idx = np.random.choice(self.n_clients, n_participants, replace=False)
        sampled_clients_idx = np.sort(sampled_clients_idx)
        return sampled_clients_idx
    
    def dispatch(self):
        for cidx in range(self.n_clients):
            self.clients[cidx].model.load_state_dict(self.global_model.state_dict())
    
    def aggregate(self, sampled_clients, **kwargs):
        new_state_dict = self.aggregator([self.clients[cidx].model.state_dict() for cidx in sampled_clients],
                                         args=self.args,
                                         sampled_clients=sampled_clients,)
        self.global_model.load_state_dict(new_state_dict)

    def global_test(self, r):
        correct, loss = 0, 0
        self.global_model = self.global_model.to(self.device)
        self.global_model.eval()
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == target).sum().item()
                loss_ = self.criterion(outputs, target)
                loss += loss_.item()
        print(f'ROUND:{r+1:>03} Global Accuracy: {100*correct/len(self.testset):.2f}%')
        print(f'ROUND:{r+1:>03}     Global Loss: {loss/len(self.testloader):.4f}')
        self.global_model = self.global_model.to('cpu')
        self.tb_update(r+1, global_acc=100*correct/len(self.testset), global_loss=loss/len(self.testloader))

    def run(self, save_period=None):
        for round in range(self.args.rounds):
            sampled_clients = self.sample_clients(int(self.args.p_ratio*self.n_clients))
            for client in sampled_clients:
                self.client_train(client)
                loss, acc = self.client_test(client)
            self.aggregate(sampled_clients)
            self.global_test(r=round)
            
            if save_period and (round+1) % save_period == 0:
                self.save_global_model(round+1)
            self.dispatch()
            self.round += 1 # TODO: FIX THIS
            print(f"####### ROUND {round+1} END #######\n")

    def client_train(self, client_id):
        self.clients[client_id].train()

    def client_test(self, client_id):
        loss, acc = self.clients[client_id].test()
        return loss, acc
        
    def save_global_model(self, round):
        torch.save(self.global_model.state_dict(), f'{self.save_path }/global_{round}.pth')

if __name__ == '__main__':    
    pass