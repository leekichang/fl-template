import io
import os
import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import models
import config as cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name'    , help='experiement name' , type=str     , default='Template')
    parser.add_argument('--model'       , help='model'            , type=str     , default='MLP')
    parser.add_argument('--dataset'     , help='dataset'          , type=str     , default='MNIST')
    parser.add_argument('--optimizer'   , help='optimizer'        , type=str     , default='SGD')
    parser.add_argument('--lr'          , help='learning rate'    , type=float   , default=1e-3)
    parser.add_argument('--decay'       , help='weight decay'     , type=float   , default=1e-4)
    parser.add_argument('--batch_size'  , help='batch size'       , type=int     , default=64)
    parser.add_argument('--seed'        , help='random seed'      , type=int     , default=0)
    parser.add_argument('--epoch'       , help='number of epochs' , type=int     , default=5)
    parser.add_argument('--use_tb'      , help='use tensorboard'  , type=str2bool, default=False)
    ### FL PARAMS ###
    parser.add_argument('--n_clients'   , help='number of clients', type=int     , default=10)
    parser.add_argument('--rounds'      , help='number of clients', type=int     , default=100)
    parser.add_argument('--alpha'       , help='Dirichlet alpha'  , type=float   , default=0.5)
    parser.add_argument('--p_ratio'     , help='Participant ratio', type=float   , default=0.2)
    parser.add_argument('--client'      , help='Client type'      , type=str     , default='NaiveClient')
    parser.add_argument('--aggregator'  , help='Server aggregator', type=str     , default='FedAvg')
    args = parser.parse_args()
    return args
    
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_model(args):
    return getattr(models, args.model)(n_class=cfg.N_CLASS[args.dataset])  

def build_criterion(args):
    return getattr(torch.nn, 'CrossEntropyLoss')()

def build_optimizer(model, args):
    return getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.decay)

def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
def print_info(args):
    def print_section(title, content):
        print(f"\n<{title}>")
        print("="*20)
        for name, value in content:
            print(f"{name:<20} {value}")
        print("="*20)

    experiment_params = [
        ('Experiment Name:', args.exp_name),
        ('Model:', args.model),
        ('Dataset:', args.dataset),
        ('Optimizer:', args.optimizer),
        ('Learning Rate:', args.lr),
        ('Weight Decay:', args.decay),
        ('Batch Size:', args.batch_size),
        ('Random Seed:', args.seed),
        ('Number of Epochs:', args.epoch),
        ('Use Tensorboard:', args.use_tb),
    ]
    
    fl_params = [
        ('Number of Clients:', args.n_clients),
        ('Number of Rounds:', args.rounds),
        ('Dirichlet Alpha:', args.alpha),
        ('Participant Ratio:', args.p_ratio),
        ('Client type:', args.client),
        ('Server Aggregator:', args.aggregator),
    ]
    print_section("Experiment Configuration", experiment_params)
    print_section("Federated Learning Parameters", fl_params)

    
def format_args(args):
    return "|".join([
        args.exp_name,
        args.model,
        args.dataset,
        args.optimizer,
        str(args.lr),
        str(args.decay),
        str(args.batch_size),
        str(args.seed),
        str(args.epoch),
        str(args.use_tb),
        str(args.n_clients),
        str(args.rounds),
        str(args.alpha),
        str(args.p_ratio),
        args.client,
        args.aggregator,
    ])

