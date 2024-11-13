import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv, APPNPConv, GATConv


class MyMLP(nn.Module):
    """MLP with node-wise ensemble coefficients learning"""
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            num_views,
            rank,
            tanh,
            norm_type="none",
    ):
        super(MyMLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.projector = nn.Linear(hidden_dim, rank, bias=False)
        self.cols = nn.Parameter(torch.ones((rank, num_views)) / num_views)

        if tanh:
            self.tanh = torch.tanh
        else:
            self.tanh = lambda x: x

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)

        coefs = self.tanh(self.projector(h_list[-1]))
        coefs =  F.softmax(coefs @ self.cols)
        return None, h, coefs


class MLP(nn.Module):
    def __init__(
            self,
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_ratio,
            norm_type="none",
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h_list, h


class RSAGELayer(nn.Module):
    def __init__(self, views, in_dim, out_dim, activation=None):
        super(RSAGELayer, self).__init__()
        # One GNN layer for each view adjacency matrix
        self.layers = nn.ModuleDict({
            view: SAGEConv(
                in_dim,
                out_dim,
                aggregator_type='gcn',
                activation=activation,
            ) for view in views
        })

    def forward(self, Gs, X, return_embeddings=False):
        view_embeddings = {
            view: self.layers[view](G, X) for view, G in Gs.items()
        }

        embedding = torch.stack(
            list(view_embeddings.values())
        ).mean(dim=0)
        if return_embeddings:
            return embedding, view_embeddings
        return embedding


class RSAGE(nn.Module):
    def __init__(self, views, in_dim, hid_dim, out_dim, dropout):
        super(RSAGE, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = RSAGELayer(views, in_dim, hid_dim, activation=F.relu)
        self.conv2 = RSAGELayer(views, hid_dim, out_dim)

    def forward(self, Gs, X, return_embeddings=False):
        X = self.dropout(self.conv1(Gs, X))
        return self.conv2(Gs, X, return_embeddings)


class RGCNLayer(nn.Module):
    def __init__(self, views, in_dim, out_dim, activation=None):
        super(RGCNLayer, self).__init__()
        # One GNN layer for each view adjacency matrix
        self.layers = nn.ModuleDict({
            view: GraphConv(
                in_dim,
                out_dim,
                activation=activation,
            ) for view in views
        })

    def forward(self, Gs, X, return_embeddings=False):
        view_embeddings = {
            view: self.layers[view](G, X) for view, G in Gs.items()
        }

        embedding = torch.stack(
            list(view_embeddings.values())
        ).mean(dim=0)
        if return_embeddings:
            return embedding, view_embeddings
        return embedding


class RGCN(nn.Module):
    def __init__(self, views, in_dim, hid_dim, out_dim, dropout):
        super(RGCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = RGCNLayer(views, in_dim, hid_dim, activation=F.relu)
        self.conv2 = RGCNLayer(views, hid_dim, out_dim)

    def forward(self, Gs, X, return_embeddings=False):
        X = self.dropout(self.conv1(Gs, X))
        return self.conv2(Gs, X, return_embeddings)


class RGATLayer(nn.Module):
    def __init__(self, views, in_dim, out_dim, layer_num_heads,
                 feat_drop, attn_drop, activation=None):
        super(RGATLayer, self).__init__()
        # One GNN layer for each view adjacency matrix
        self.layers = nn.ModuleDict({
            view: GATConv(
                in_dim,
                out_dim,
                layer_num_heads,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                activation=activation,
            ) for view in views
        })

    def forward(self, Gs, X, return_embeddings=False):
        view_embeddings = {
            view: self.layers[view](G, X).flatten(1) for view, G in Gs.items()
        }

        embedding = torch.stack(
            list(view_embeddings.values())
        ).mean(dim=0)
        if return_embeddings:
            return embedding, view_embeddings
        return embedding


class RGAT(nn.Module):
    def __init__(self, views, in_dim, hid_dim, out_dim, heads, feat_drop, attn_drop):
        super(RGAT, self).__init__()
        self.conv1 = RGATLayer(views, in_dim, hid_dim, heads[0], feat_drop, attn_drop, activation=F.relu)
        self.conv2 = RGATLayer(views, hid_dim * heads[0], out_dim, heads[1], feat_drop, attn_drop)

    def forward(self, Gs, X, return_embeddings=False):
        X = self.conv1(Gs, X)
        return self.conv2(Gs, X, return_embeddings)


class RAPPNP(nn.Module):
    def __init__(self, views, in_dim, hid_dim, out_dim, feat_drop, edge_drop=0.5, alpha=0.1, k=10):
        super(RAPPNP, self).__init__()
        self.mlps = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(feat_drop),
            nn.Linear(hid_dim, out_dim),
        )
        self.convs = nn.ModuleDict({
            view: APPNPConv(k, alpha, edge_drop) for view in views
        })

    def forward(self, Gs, X):
        X = self.mlps(X)
        view_preds = []
        for view, G in Gs.items():
            view_preds.append(self.convs[view](G, X))
        preds = torch.stack(
            view_preds
        ).mean(dim=0)

        return preds


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    def __init__(self, views, in_dim, out_dim, layer_num_heads,
                 feat_drop, attn_drop, activation=None):
        super(HANLayer, self).__init__()

        # One GAT layer for each view adjacency matrix
        self.layers = nn.ModuleDict({
            view: GATConv(
                in_dim,
                out_dim,
                layer_num_heads,
                feat_drop,
                attn_drop,
                activation=activation,
            ) for view in views
        })

        self.semantic_attention = SemanticAttention(
            in_size=out_dim * layer_num_heads
        )

    def forward(self, Gs, X, return_embeddings=False):
        view_embeddings = {
            view: self.layers[view](G, X).flatten(1) for view, G in Gs.items()
        }

        embedding = torch.stack(
            list(view_embeddings.values()), dim=1
        )  # (N, M, D * K)
        embedding = self.semantic_attention(embedding)  # (N, D * K)

        if return_embeddings:
            return embedding, view_embeddings
        return embedding


class HAN(nn.Module):
    def __init__(
            self, views, in_dim, hid_dim, out_dim, heads, feat_drop, attn_drop
    ):
        super(HAN, self).__init__()
        self.conv1 = HANLayer(views, in_dim, hid_dim, heads[0], feat_drop, attn_drop, activation=F.relu)
        self.conv2 = HANLayer(views, hid_dim * heads[0], out_dim, heads[1], feat_drop, attn_drop)

    def forward(self, Gs, X, return_embeddings=False):
        X = self.conv1(Gs, X)
        return self.conv2(Gs, X, return_embeddings=return_embeddings)


class Model(nn.Module):
    """
    Wrapper of different models
    """

    def __init__(self, conf):
        super(Model, self).__init__()
        self.model_name = conf["model_name"]
        if "MyMLP" in conf["model_name"]:
            self.encoder = MyMLP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                num_views=conf['num_views'],
                rank=conf['rank'],
                tanh=conf['tanh'],
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "MLP" in conf["model_name"]:
            self.encoder = MLP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "SAGE" in conf["model_name"]:
            self.encoder = RSAGE(
                views=conf["views"],
                in_dim=conf["feat_dim"],
                hid_dim=conf["hidden_dim"],
                out_dim=conf["label_dim"],
                dropout=conf["dropout_ratio"]
            ).to(conf["device"])
        elif "GCN" in conf["model_name"]:
            self.encoder = RGCN(
                views=conf["views"],
                in_dim=conf["feat_dim"],
                hid_dim=conf["hidden_dim"],
                out_dim=conf["label_dim"],
                dropout=conf["dropout_ratio"]
            ).to(conf["device"])
        elif "GAT" in conf["model_name"]:
            self.encoder = RGAT(
                views=conf["views"],
                in_dim=conf["feat_dim"],
                hid_dim=conf["hidden_dim"],
                out_dim=conf["label_dim"],
                heads=(conf["num_heads"], 1),
                feat_drop=conf["dropout_ratio"],
                attn_drop=conf["attn_dropout_ratio"]
            ).to(conf["device"])
        elif "HAN" in conf["model_name"]:
            self.encoder = HAN(
                views=conf["views"],
                in_dim=conf["feat_dim"],
                hid_dim=conf["hidden_dim"],
                out_dim=conf["label_dim"],
                heads=(conf["num_heads"], 1),
                feat_drop=conf["dropout_ratio"],
                attn_drop=conf["attn_dropout_ratio"]
            ).to(conf["device"])
        elif "APPNP" in conf["model_name"]:
            self.encoder = RAPPNP(
                views=conf["views"],
                in_dim=conf["feat_dim"],
                hid_dim=conf["hidden_dim"],
                out_dim=conf["label_dim"],
                feat_drop=conf["dropout_ratio"]
            ).to(conf["device"])

    def forward(self, data, feats):
        """
        data: a graph `g` or a `dataloader` of blocks
        """
        if "MLP" in self.model_name:
            return self.encoder(feats)
        else:
            return self.encoder(data, feats)

    # def forward_fitnet(self, data, feats):
    #     """
    #     Return a tuple (h_list, h)
    #     h_list: intermediate hidden representation
    #     h: final output
    #     """
    #     if "MLP" in self.model_name:
    #         return self.encoder(feats)
    #     else:
    #         return self.encoder(data, feats)

    def inference(self, data, feats, return_embeddings=False):
        if "MLP" in self.model_name:
            return self.encoder(feats)[1]
        else:
            return self.encoder(data, feats, return_embeddings)
