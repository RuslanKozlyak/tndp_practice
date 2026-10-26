"""Нейросетевые головы оценки маршрута и HALT."""
import torch
from torch import nn
from tndp.environment.features import halt_features
from tndp.models.layers import DEFAULT_PREFERENCE_GOALS, FeatureNorm, get_mlp

class RouteScorer(nn.Module):
    # Feature spec: order of the cost-weight slots inside global state
    # features, locked to the trained checkpoints (see DEFAULT_PREFERENCE_GOALS).
    preference_goals = DEFAULT_PREFERENCE_GOALS

    def __init__(self, embed_dim, nonlin_type, dropout, n_mlp_layers=2,
                 mlp_width=None, n_extra_feats=14, preference_goals=None):
        super().__init__()
        if preference_goals is not None:
            self.preference_goals = tuple(preference_goals)
        self.embed_dim = embed_dim
        # route_feats (2) + global_state_features. Default 14 = 2 + 12; the edit
        # model with adjustment conditioning passes 16 = 2 + 14.
        self.n_extra_feats = n_extra_feats
        self.dropout = dropout
        if mlp_width is None:
            mlp_width = self.in_dim * 2
        self.out_mlp = get_mlp(n_mlp_layers, mlp_width, nonlin_type,
                               dropout, in_dim=self.in_dim, out_dim=1)
        self.extras_norm = FeatureNorm(self.n_extra_feats)

    @property
    def in_dim(self):
        return self.embed_dim * 2 + self.n_extra_feats

    def forward(self, state, node_descs, route_idxs, route_time,
                node_padding_mask=None):
        # assemble route sequences
        route_gather_idxs = route_idxs * (route_idxs > -1)
        route_gather_idxs = \
            route_gather_idxs.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        route_seqs = node_descs.gather(1, route_gather_idxs)

        route_len = (route_idxs > -1).sum(dim=1, dtype=torch.float)
        extra_feats = halt_features(state, route_time, route_len)

        # pass sequences through encoder
        route_desc = self.encode(node_descs, route_seqs, node_padding_mask,
                                 route_idxs == -1)

        global_node_desc = node_descs.mean(dim=1)

        # normalize extra features
        extra_feats = self.extras_norm(extra_feats)
        # pass encoding and features through MLP
        in_vec = torch.cat((route_desc,
                            global_node_desc,
                            extra_feats), dim=1)
        scores = self.out_mlp(in_vec)
        return scores

    def encode(self, node_descs, route_seqs, node_pad_mask=None,
               route_pad_mask=None, embed_pos=False):
        raise NotImplementedError

class RouteEndpointsScorer(RouteScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        placeholder = torch.zeros((1, 1, self.embed_dim * 2),
                                  dtype=torch.float32)
        self.register_buffer('placeholder', placeholder)

    @property
    def in_dim(self):
        return self.n_extra_feats + self.embed_dim * 3

    def encode(self, node_descs, route_seqs, node_pad_mask, route_pad_mask):
        # get first and last node in each route sequence
        first_nodes = route_seqs[:, 0]
        idxs = (~route_pad_mask).sum(dim=1, keepdim=True) - 1
        route_is_empty = route_pad_mask[:, 0]
        # negative indices will occur if the route is empty
        idxs[route_is_empty] = 0
        idxs = idxs[..., None].expand(-1, -1, route_seqs.shape[-1])
        last_nodes = route_seqs.gather(1, idxs).squeeze(1)
        descs = torch.cat((first_nodes, last_nodes), dim=-1)
        # where the route is empty, use a placeholder
        descs[route_is_empty] = self.placeholder
        # append the mean descriptor of the nodes
        # mean_descs = mean_pool_sequence(node_descs, node_pad_mask)
        # descs = torch.cat((descs, mean_descs), dim=-1)
        return descs
