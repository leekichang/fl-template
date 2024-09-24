import os
import sys
import copy
import torch
import numpy as np

class Dirichlet(object):
    def __init__(self, trainset, testset, n_clients, alpha=0.5, min_samples=10):
        self.n_clients = n_clients
        self.alpha = alpha
        self.trainset = trainset
        self.testset  = testset
        self.num_classes = self.testset.targets.max().item()+1
        self.total_train_samples = len(self.trainset)
        self.total_test_samples  = len(self.testset)
        self.min_samples = min_samples
        
    def split_dataset(self):
        # TODO: Need to implement minimum number of samples per client
        dirichlet_dist = np.random.dirichlet([self.alpha] * self.n_clients, self.num_classes)
        grouped_data_train = [[] for _ in range(self.n_clients)]
        grouped_data_test  = [[] for _ in range(self.n_clients)]
        
        for label in range(self.num_classes):
            train_label_indices = np.where(self.trainset.targets == label)[0]
            test_label_indices  = np.where(self.testset.targets == label)[0]
            np.random.shuffle(train_label_indices)
            np.random.shuffle(test_label_indices)
            
            current_train_idx, current_test_idx = 0, 0
            remaining_samples_train = len(train_label_indices) - self.min_samples * self.n_clients
            remaining_samples_test = len(test_label_indices) - (self.min_samples * self.n_clients * self.total_test_samples // self.total_train_samples)
            for cidx in range(self.n_clients):
                num_samples_train = self.min_samples + int(dirichlet_dist[label, cidx] * remaining_samples_train)
                grouped_data_train[cidx].extend(train_label_indices[current_train_idx:current_train_idx + num_samples_train])
                current_train_idx += num_samples_train
                
                num_samples_test = self.min_samples * self.total_test_samples // self.total_train_samples + int(dirichlet_dist[label, cidx] * remaining_samples_test)
                grouped_data_test[cidx].extend(test_label_indices[current_test_idx:current_test_idx + num_samples_test])
                current_test_idx += num_samples_test

                # num_samples = int(dirichlet_dist[label, cidx] * len(train_label_indices))
                # grouped_data_train[cidx].extend(train_label_indices[current_train_idx:current_train_idx + num_samples])
                # current_train_idx += num_samples
                
                # num_samples = num_samples*self.total_test_samples//self.total_train_samples
                # grouped_data_test[cidx].extend(test_label_indices[current_test_idx:current_test_idx + num_samples])
                # current_test_idx += num_samples
                
        grouped_data_trainsets = [copy.deepcopy(self.trainset) for _ in range(self.n_clients)]
        grouped_data_testsets = [copy.deepcopy(self.testset) for _ in range(self.n_clients)]
        
        for cidx in range(self.n_clients):
            indices = grouped_data_train[cidx]
            grouped_data_trainsets[cidx].data    = copy.deepcopy(self.trainset.data[indices])
            grouped_data_trainsets[cidx].targets = copy.deepcopy(self.trainset.targets[indices])
            
            indices = grouped_data_test[cidx]
            grouped_data_testsets[cidx].data    = copy.deepcopy(self.testset.data[indices])
            grouped_data_testsets[cidx].targets = copy.deepcopy(self.testset.targets[indices])
        
        return grouped_data_trainsets, grouped_data_testsets
        # for i, (dst) in enumerate(grouped_datasets):
        #     images, labels = dst.data, dst.targets
        #     print(f'Group {i + 1}: {len(images)} samples')
        #     print(labels)
        #     for j in range(10):
        #         print(f'  Label {j}: {torch.sum(labels==j)} samples')
    
if __name__ == '__main__':
    import DataManager.datamanager as dm
    trainset, testset = dm.MNIST()
    dirichlet = Dirichlet(trainset, 5, 0.5)
    dirichlet.split_dataset()
    