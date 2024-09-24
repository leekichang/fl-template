import sys
import torch
import numpy as np

class FedAvg(object):
    def __init__(self):
        pass

    def __call__(self, client_state_dicts, **kwargs):
        new_state_dict = {}

        param_lists = {k: [] for k in client_state_dicts[0].keys()}

        for client_state_dict in client_state_dicts:
            for k in param_lists.keys():
                param_lists[k].append(client_state_dict[k].cpu().numpy())

        for k in param_lists.keys():
            param_array = np.array(param_lists[k])
            mean_param = torch.tensor(np.mean(param_array, axis=0), dtype=torch.float32)
            new_state_dict[k] = mean_param

        return new_state_dict

class Median(object):
    def __init__(self):
        pass

    def __call__(self, client_state_dicts, **kwargs):
        new_state_dict = {}

        param_lists = {k: [] for k in client_state_dicts[0].keys()}

        for client_state_dict in client_state_dicts:
            for k in param_lists.keys():
                param_lists[k].append(client_state_dict[k].cpu().numpy())

        for k in param_lists.keys():
            param_array = np.array(param_lists[k])
            median_param = torch.tensor(np.median(param_array, axis=0), dtype=torch.float32)
            new_state_dict[k] = median_param

        return new_state_dict

class TrimmedMean(object):
    def __init__(self):
        pass

    def __call__(self, client_state_dicts, **kwargs):
        if 'args' in kwargs:
            args = kwargs['args']
        if 'trim_fraction' in kwargs:
            trim_fraction = kwargs['trim_fraction']
        else:
            trim_fraction = 0.2
            
        new_state_dict = {}
        param_lists = {k: [] for k in client_state_dicts[0].keys()}

        for client_state_dict in client_state_dicts:
            for k in param_lists.keys():
                param_lists[k].append(client_state_dict[k].cpu().numpy())

        for k in param_lists.keys():
            param_array = np.array(param_lists[k])
            n_trim = int(trim_fraction * param_array.shape[0])
            sorted_array = np.sort(param_array, axis=0)
            trimmed_array = sorted_array if n_trim == 0 else sorted_array[n_trim: -n_trim]
            trimmed_mean_param = torch.tensor(np.mean(trimmed_array, axis=0), dtype=torch.float32)
            new_state_dict[k] = trimmed_mean_param
        return new_state_dict

class Krum(object):
    def __init__(self):
        pass
    
    def euclidean_distance(self, w1, w2):
        dist = 0
        for k in w1.keys():
            dist += np.linalg.norm(w1[k] - w2[k])
        return dist
    
    def __call__(self, client_state_dicts, **kwargs):
        if 'args' in kwargs:
            args = kwargs['args']
        if 'n_attackers' in kwargs:
            n_attackers = kwargs['n_attackers']
        else:
            n_attackers = 1

        num_clients = len(client_state_dicts)
        dist_matrix = np.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                dist = self.euclidean_distance(client_state_dicts[i], client_state_dicts[j])
                dist_matrix[i,j]=dist
                dist_matrix[j,i]=dist
        min_sum_dist = float('inf')
        selected_index = -1
        for i in range(num_clients):
            sorted_indices = np.argsort(dist_matrix[i])
            sum_dist = np.sum(dist_matrix[i, sorted_indices[1:(num_clients - n_attackers)]]) # exclude itself
            if sum_dist < min_sum_dist:
                min_sum_dist = sum_dist
                selected_index = i
        return client_state_dicts[selected_index]