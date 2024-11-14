import dgl
import torch
import pickle
import numpy as np
from pathlib import Path
import scipy.sparse as sp


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.asarray(features.sum(1))
    mask = np.equal(rowsum, 0.0).flatten()
    rowsum[mask] = np.nan
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[mask] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.asarray(features.todense())


def build_graph_from_edges(adj, num_nodes)-> dgl.DGLGraph:
    src = torch.from_numpy(adj[0])
    dst = torch.from_numpy(adj[1])
    # src = torch.tensor(src, dtype=torch.int32)
    # dst = torch.tensor(dst, dtype=torch.int32)
    G = dgl.graph((src, dst), num_nodes=num_nodes, idtype=torch.int64)
    return dgl.add_self_loop(dgl.to_bidirected(G))


def load_data(dataset, dataset_path):
    datasets1 = ['ACM', 'IMDB', 'IMDB5K', 'DBLP']
    datasets2 = ['ArXiv', 'MAG']
    assert dataset in (datasets1 + datasets2)

    data_path = Path.cwd().joinpath(dataset_path, f'{dataset.lower()}.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    X = data['feats']
    if dataset in datasets1:
        X = preprocess_features(X)
        X = torch.tensor(X, dtype=torch.float32)
    elif dataset in datasets2:
        X = torch.tensor(X, dtype=torch.float32)

    Y = torch.tensor(data['labels'], dtype=torch.int64)

    adjs = data['adjs']
    num_nodes = Y.shape[0]
    Gs = {view: build_graph_from_edges(adj, num_nodes)
          for view, adj in adjs.items()}

    splits = data['splits']
    idx_train = torch.from_numpy(splits['train_mask'].nonzero()[0])
    idx_val = torch.from_numpy(splits['val_mask'].nonzero()[0])
    idx_test = torch.from_numpy(splits['test_mask'].nonzero()[0])

    print(f'Load {dataset} dataset... \n'
          f'It has {X.shape[0]} nodes, {X.shape[1]}-dim input features, \n'
          f'and {[G.num_edges() for G in Gs.values()]} edges for views {list(Gs.keys())}. \n'
          f'The train/val/test is {idx_train.shape[0]}/{idx_val.shape[0]}/{idx_test.shape[0]}')

    return Gs, X, Y, idx_train, idx_val, idx_test
