
import torch
from torch.utils.data import DataLoader, TensorDataset, Sampler
import numpy as np
import torch.nn.functional as F
from scipy.spatial import distance
import torch.nn as nn

def linear_block(input_size, output_size, batch_norm=True, dropout_prob=0.3):
    layers = [nn.Linear(input_size, output_size)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(output_size))
    layers.append(nn.ReLU())
    if dropout_prob > 0:
        layers.append(nn.Dropout(dropout_prob))
    return layers

class OpcodeEmbedding(nn.Module): # Can revise
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.3):
        super(OpcodeEmbedding, self).__init__()
        self.layers = nn.Sequential(
            *linear_block(input_size, hidden_size, batch_norm=True, dropout_prob=dropout_prob),
            *linear_block(hidden_size, hidden_size, batch_norm=True, dropout_prob=dropout_prob),
            *linear_block(hidden_size, output_size, batch_norm=False, dropout_prob=0)
        )


    def forward(self, x):
        return self.layers(x)
    
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, dropout_rate=0.5):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction


class PrototypeNet(nn.Module):
    def __init__(self, embedding_model):
        super(PrototypeNet, self).__init__()
        self.embedding_model = embedding_model

    def forward(self, x):
        x = self.embedding_model(x)
        return x.view(x.size(0), -1)
    
def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def prototypical_loss(input, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    _, y_hat = log_p_y.max(2)
    # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    log_p_y = log_p_y.view(-1, n_classes)
    target_ = target_inds.reshape(-1).squeeze()
    # print(log_p_y.shape, target_.shape)
    nll_loss = nn.NLLLoss()
    loss_val = nll_loss(log_p_y, target_)
    
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
    
    return loss_val,  acc_val, prototypes, classes, query_samples, target_inds, y_hat

def prototypical_loss_with_cross_entropy(input, target, n_support, ce_weight=0.5):
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)
    
    # 原有的 prototypical loss 計算
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    proto_loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    
    # 計算 cross entropy loss
    ce_target = torch.arange(n_classes).repeat_interleave(n_query)
    ce_loss = F.cross_entropy(-dists, ce_target)
    
    # 結合兩種 loss
    combined_loss = (1 - ce_weight) * proto_loss + ce_weight * ce_loss
    
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
    
    return combined_loss, acc_val, prototypes, classes, query_samples, target_inds, y_hat


def nn_prototypical_loss(input, target, n_support, ce_weight=0.5, margin=2.0):
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    # 每個 support sample 都是一個 prototype
    prototypes = torch.cat([input_cpu[idx_list] for idx_list in support_idxs])

    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = input.to('cpu')[query_idxs]

    # 計算查詢樣本和所有原型之間的歐氏距離
    dists = euclidean_dist(query_samples, prototypes)
    # 找出每個查詢樣本與每個類別中最近的原型的距離
    dists_by_class = dists.view(n_classes * n_query, n_classes, n_support)
    closest_dists, _ = dists_by_class.min(dim=2)

    log_p_y = F.log_softmax(-closest_dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    _, y_hat = log_p_y.max(2)
    # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    log_p_y = log_p_y.view(-1, n_classes)
    target_ = target_inds.reshape(-1).squeeze()
    nll_loss = nn.NLLLoss()
    loss_val = nll_loss(log_p_y, target_)
    
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val, acc_val, prototypes, classes, query_samples, target_inds, y_hat


def contrastive_loss(input, target, n_support, margin=2.0):
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = input.to('cpu')[query_idxs]

    # 計算查詢樣本和原型之間的歐氏距離
    dists = euclidean_dist(query_samples, prototypes)

    # 創建標籤矩陣
    labels = torch.zeros_like(dists)
    for i, c in enumerate(classes):
        labels[i * n_query : (i + 1) * n_query, i] = 1
    # 計算 contrastive loss
    positive_loss = (labels * dists.pow(2)).sum(1)
    negative_loss = ((1 - labels) * F.relu(margin - dists).pow(2)).sum(1)
    loss_val = (positive_loss + negative_loss).mean()

    # 計算準確率
    _, y_hat = dists.min(1)
    target_inds = torch.arange(0, n_classes).repeat_interleave(n_query)
    acc_val = y_hat.eq(target_inds).float().mean()

    return loss_val, acc_val, prototypes, classes, query_samples, target_inds, y_hat

def prototypical_loss_using_proto(input, target, prototypes, label_mapping = None):
    # 修改原本的model使其不使用testing data計算prototypes
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    # prototypes is dict
    prototypes_torch = torch.tensor([])
    for key in prototypes:
        prototypes_torch = torch.cat((prototypes_torch, prototypes[key].unsqueeze(0)), 0)
    dists = euclidean_dist(input_cpu, prototypes_torch)
    log_p_y = F.log_softmax(-dists, dim=1)
    # loss_val = -log_p_y.gather(1, target_cpu.view(-1, 1)).squeeze().mean()

    _, y_hat = log_p_y.max(1)  
 
    for i in range(y_hat.shape[0]):
        for k, key in enumerate(label_mapping):
            if y_hat[i] in label_mapping[key]:
                y_hat[i] = k

    acc_val = y_hat.eq(target_cpu).float().mean()

    return acc_val, y_hat


def get_prototypes(model, dataloader, support_shots):
    '''
    Get the prototypes for each class
    '''
    model.eval()
    prototypes = dict()
    prototypesCounts = dict()
    for idx, (data, labels) in enumerate(dataloader):
        data = data.squeeze(1)
        data = data.float()
        model_output = model(data)
        _, _, proto, protoID, _, _, _ = prototypical_loss(model_output, target=labels, n_support=support_shots)

        for i in range(proto.shape[0]):
            if prototypesCounts.get(protoID[i].item()) is None:
                prototypesCounts[protoID[i].item()] = 1
                prototypes[protoID[i].item()] = proto[i]
            else:
                prototypesCounts[protoID[i].item()] += 1
                prototypes[protoID[i].item()] += proto[i]
    prototypes = {k: v / prototypesCounts[k] for k, v in prototypes.items()}
    # sort the prototypes by key
    prototypes = dict(sorted(prototypes.items()))
    print(prototypes.keys())
    return prototypes

# def prototypical_loss_test(input, target, n_support):
#     target_cpu = target.to('cpu')
#     input_cpu = input.to('cpu')

#     def supp_idxs(c):
#         return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

#     classes = torch.unique(target_cpu)
#     n_classes = len(classes)

#     support_idxs = list(map(supp_idxs, classes))

#     prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

#     query_idxs = []
#     n_queries = []
#     for c in classes:
#         class_idxs = target_cpu.eq(c).nonzero().squeeze(1)
#         n_query = len(class_idxs) - n_support
#         n_queries.append(n_query)
#         query_idxs.extend(class_idxs[n_support:])
#     # print(query_idxs)
#     query_samples = input_cpu[query_idxs]
#     dists = euclidean_dist(query_samples, prototypes)

#     log_p_y = F.log_softmax(-dists, dim=1)

#     target_inds = torch.cat([torch.full((n_query,), i, dtype=torch.long) for i, n_query in enumerate(n_queries)])

#     loss_val = -log_p_y.gather(1, target_inds.unsqueeze(1)).squeeze().mean()
#     _, y_hat = log_p_y.max(1)
#     acc_val = y_hat.eq(target_inds).float().mean()
    
#     return loss_val, acc_val, prototypes, classes, query_samples, target_inds, y_hat

def get_original_prototype(dataNp: np.array, clusterLabel: np.array) -> np.array:
	unique = np.unique(clusterLabel)
	for i, label in enumerate(unique):
		clusterData = dataNp[clusterLabel == label]
		prototype = np.mean(clusterData, axis = 0)
		prototype = prototype.reshape(1, -1)
		if i == 0:
			originalPrototype = prototype
		else:
			originalPrototype = np.concatenate((originalPrototype, prototype), axis = 0)
	return originalPrototype