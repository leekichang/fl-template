import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")

def MNIST():
    MNIST_train  = datasets.MNIST(root='../../../disk1/Kichang/dataset', train=True,  download=False, transform=transforms.ToTensor())
    MNIST_test   = datasets.MNIST(root='../../../disk1/Kichang/dataset', train=False, download=False, transform=transforms.ToTensor())
    return MNIST_train, MNIST_test

def EMNIST(split='balanced'):
    EMNIST_train = datasets.EMNIST(root='../../../disk1/Kichang/dataset', split=split, train=True , download=True, transform=transforms.ToTensor())
    EMNIST_test  = datasets.EMNIST(root='../../../disk1/Kichang/dataset', split=split, train=False, download=True, transform=transforms.ToTensor())
    return EMNIST_train, EMNIST_test

def FashionMNIST():
    FashionMNIST_train = datasets.FashionMNIST(root='../../../disk1/Kichang/dataset', train=True , download=False, transform=transforms.ToTensor())
    FashionMNIST_test  = datasets.FashionMNIST(root='../../../disk1/Kichang/dataset', train=False, download=False, transform=transforms.ToTensor())
    return FashionMNIST_train, FashionMNIST_test

def CIFAR10():
    mean = [0.49139968, 0.48215841, 0.44653091]
    std = [0.24703223, 0.24348513, 0.26158784]
    transform  = transforms.Compose([transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std, inplace=True)])
    CIFAR10_train = datasets.CIFAR10(root='../../../disk1/Kichang/dataset', train=True , download=False, transform=transform)
    transform  = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean, std, inplace=True)])
    CIFAR10_test  = datasets.CIFAR10(root='../../../disk1/Kichang/dataset', train=False, download=False, transform=transform)
    CIFAR10_train.targets = torch.LongTensor(CIFAR10_train.targets)
    CIFAR10_test.targets  = torch.LongTensor(CIFAR10_test.targets)
    return CIFAR10_train, CIFAR10_test

def CIFAR100():
    mean = [0.50707516, 0.48654887, 0.44091784]
    std  = [0.26733429, 0.25643846, 0.27615047]
    transform  = transforms.Compose([transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std, inplace=True)])
    CIFAR100_train = datasets.CIFAR100(root='../../../disk1/Kichang/dataset', train=True , download=False, transform=transform)
    transform  = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean, std, inplace=True)])
    CIFAR100_test  = datasets.CIFAR100(root='../../../disk1/Kichang/dataset', train=False, download=False, transform=transform)
    CIFAR100_train.targets = torch.LongTensor(CIFAR100_train.targets)
    CIFAR100_test.targets  = torch.LongTensor(CIFAR100_test.targets)
    return CIFAR100_train, CIFAR100_test

if __name__ == '__main__':
    EMNIST_train = datasets.EMNIST(root='../../../disk1/Kichang/dataset', split='balanced', train=True , download=True, transform=transforms.ToTensor())
    EMNIST_test  = datasets.EMNIST(root='../../../disk1/Kichang/dataset', split='balanced', train=False, download=True, transform=transforms.ToTensor())
    print(max(EMNIST_train.targets))
