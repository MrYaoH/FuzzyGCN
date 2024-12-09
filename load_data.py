import numpy as np
import torch

from torch_geometric import datasets
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
# from wiki_process import WikiCS  ###

import scipy.sparse as sp


def get_dataset(root, name, transform=NormalizeFeatures()):
    pyg_dataset_dict = {
        'coauthor-cs': (datasets.Coauthor, 'CS'),
        'coauthor-physics': (datasets.Coauthor, 'physics'),
        'amazon-computers': (datasets.Amazon, 'Computers'),
        'amazon-photos': (datasets.Amazon, 'Photo'),
        'cora': (datasets.Planetoid, 'Cora'),
        'citeseer': (datasets.Planetoid, 'CiteSeer'),
        'pubmed': (datasets.Planetoid, 'PubMed'),
    }

    assert name in pyg_dataset_dict, "Dataset must be in {}".format(list(pyg_dataset_dict.keys()))

    dataset_class, name = pyg_dataset_dict[name]
    # dataset = dataset_class(root, name=name, transform=transform)
    dataset = dataset_class(root, name=name)

    return dataset


def get_wiki_cs(root, transform=NormalizeFeatures()):
    dataset = WikiCS(root, transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    return [data], np.array(data.train_mask), np.array(data.val_mask), np.array(data.test_mask)


class ConcatDataset(InMemoryDataset):
    r"""
    PyG Dataset class for merging multiple Dataset objects into one.
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        self.__indices__ = None
        self.__data_list__ = []
        for dataset in datasets:
            self.__data_list__.extend(list(dataset))
        self.data, self.slices = self.collate(self.__data_list__)


def adj_transform(data):

    n_nodes = data.x.shape[0]
    adj_index = data.edge_index.detach().cpu()
    value = np.ones(len(adj_index[0]))

    sp_adj = sp.coo_matrix((value, (adj_index[0], adj_index[1])), shape=(n_nodes, n_nodes)).tocsr()

    nodes_to_keep = torch.LongTensor(np.arange(data.x.shape[0]))

    adj_matrix = sp_adj[nodes_to_keep][:, nodes_to_keep]
    adj_matrix = preprocess_adj(adj_matrix)
    return scipy_sparse_mat_to_torch_sparse_tensor(adj_matrix)


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    sparse matrix transfers to sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
    #return sparse_to_tuple(features)


def normalized_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj_normalized = normalized_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized
    #return sparse_to_tuple(adj_normalized)
