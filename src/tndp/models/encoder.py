"""GATv2: кодирование остановок и пар в эмбеддинги."""
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, BatchNorm, GraphNorm
from tndp.models.layers import ATTN_DEFAULT_DROPOUT, DEFAULT_NONLIN, _layer_indims_helper, get_mlp

class GraphNetBase(nn.Module):
    def __init__(self, n_layers, embed_dim, in_node_dim=None, in_edge_dim=None,
                 out_dim=None, nonlin_type=DEFAULT_NONLIN,
                 dropout=ATTN_DEFAULT_DROPOUT, recurrent=False, residual=False,
                 dense=False, n_proj_layers=1, use_norm=False, layer_kwargs={}):
        """nonlin type: nonlinearity, or the string name the non-linearity to
            use."""
        super().__init__()

        # do some input checking
        assert not (residual and dense)
        assert not (recurrent and dense)

        # set up members
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.in_node_dim = embed_dim if in_node_dim is None else in_node_dim
        self.in_edge_dim = in_edge_dim
        self.out_dim = embed_dim if out_dim is None else out_dim
        self.recurrent = recurrent
        self.residual = residual
        self.dense = dense
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        if isinstance(nonlin_type, str):
            self.nonlin_type = getattr(nn, nonlin_type)
        else:
            self.nonlin_type = nonlin_type
        self.nonlin = self.nonlin_type()
        self.embed_dim = embed_dim
        self.layer_kwargs = layer_kwargs

        self.node_in_proj = nn.Identity()
        self.edge_in_proj = nn.Identity()
        self.node_out_proj = nn.Identity()
        self.edge_out_proj = nn.Identity()

        if recurrent:
            if self.in_node_dim != self.embed_dim:
                # project nodes to embedding dimension
                self.node_in_proj = nn.Linear(self.in_node_dim, self.embed_dim)
            if self.gives_edge_features and self.in_edge_dim != self.embed_dim:
                # project edges to embedding dimension
                self.edge_in_proj = nn.Linear(self.in_edge_dim, self.embed_dim)

        final_nodedim = self._get_final_nodedim()
        if self.out_dim != final_nodedim:
            # projected embedded nodes to output dimension
            self.node_out_proj = get_mlp(n_proj_layers, self.embed_dim,
                                         in_dim=final_nodedim,
                                         out_dim=self.out_dim)
            if self.gives_edge_features:
                self.edge_out_proj = get_mlp(n_proj_layers, self.embed_dim,
                                             in_dim=self._get_final_edgedim(),
                                             out_dim=self.out_dim)

        # regardless of whether the network is recurrent, we still use separate
         # normalization layers at each step
        self.use_norm = use_norm
        if use_norm:
            norm_layers = [GraphNorm(self.embed_dim)
                        for _ in range(self.n_layers - 1)]
            self.node_norm_layers = nn.ModuleList(norm_layers)
            if self.gives_edge_features:
                norm_layers = [BatchNorm(self.embed_dim)
                            for _ in range(self.n_layers - 1)]
                self.edge_norm_layers = nn.ModuleList(norm_layers)

    def _get_layer_node_indims(self):
        return _layer_indims_helper(self.in_node_dim, self.embed_dim,
                                    self.n_layers, self.dense)

    def _get_layer_edge_indims(self):
        return _layer_indims_helper(self.in_edge_dim, self.embed_dim,
                                    self.n_layers, self.dense)

    def _get_final_nodedim(self):
        indims = self._get_layer_node_indims()
        if self.dense:
            return indims[-1] + self.embed_dim
        else:
            return self.embed_dim

    def _get_final_edgedim(self):
        indims = self._get_layer_edge_indims()
        if self.dense:
            return indims[-1] + self.embed_dim
        else:
            return self.embed_dim

    @property
    def gives_edge_features(self):
        raise NotImplementedError

    def forward(self, data):
        data = self.preprocess(data)
        if self.gives_edge_features:
            output = self._forward_helper(data)
        else:
            output = self._forward_helper(data), None
        return self.postprocess(*output)

    def preprocess(self, data):
        data.x = self.node_in_proj(data.x)
        if data.edge_attr is not None:
            data.edge_attr = self.edge_in_proj(data.edge_attr)
        return data

    def postprocess(self, node_embeds, edge_embeds):
        node_embeds = self.node_out_proj(node_embeds)
        if edge_embeds is not None:
            edge_embeds = self.edge_out_proj(edge_embeds)
        if self.gives_edge_features:
            return node_embeds, edge_embeds
        else:
            return node_embeds

    def _forward_helper(self, data):
        node_descs = data.x
        edge_descs = data.edge_attr

        for li, layer in enumerate(self.layers):
            # apply the layer and graph normalization
            out = layer(node_descs, data.edge_index, edge_descs)
            if self.gives_edge_features:
                layer_nodes, layer_edges = out
            else:
                layer_nodes, layer_edges = out, None

            if li < len(self.layers) - 1:
                # this is not the last layer, so apply dropout and nonlinearity
                if self.use_norm:
                    layer_nodes = self.node_norm_layers[li](layer_nodes,
                                                            data.batch)
                layer_nodes = self.nonlin(self.dropout(layer_nodes))
                if layer_edges is not None:
                    if self.use_norm:
                        layer_edges = self.edge_norm_layers[li](layer_edges)
                    layer_edges = self.nonlin(self.dropout(layer_edges))

            # get next layer's inputs based on this layer's outputs
            if self.residual and node_descs.shape[-1] == layer_nodes.shape[-1]:
                # add the input to the output
                node_descs = node_descs + layer_nodes
                if layer_edges is not None:
                    edge_descs = edge_descs + layer_edges
            elif self.dense:
                # concatenate the input to the output
                node_descs = torch.cat((node_descs, layer_nodes), dim=-1)
                if layer_edges is not None:
                    edge_descs = torch.cat((edge_descs, layer_edges), dim=-1)
            else:
                # just replace the input with the output
                node_descs = layer_nodes
                if layer_edges is not None:
                    edge_descs = layer_edges

        assert node_descs.shape[-1] == self._get_final_nodedim()

        # return the outputs
        if not self.gives_edge_features:
            # don't return the same edge features
            return node_descs
        else:
            return node_descs, edge_descs

class GraphAttnNet(GraphNetBase):
    def __init__(self, n_heads=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_heads = n_heads
        assert self.embed_dim % n_heads == 0, \
            'embed_dim is not divisible by n_heads!'
        head_dim = self.embed_dim // self.n_heads
        make_layer = lambda in_dim: \
            GATv2Conv(in_dim, head_dim, self.n_heads,
                      edge_dim=self.in_edge_dim, **self.layer_kwargs)

        if self.recurrent:
            layers = [make_layer(self.embed_dim)] * self.n_layers
        else:
            in_dims = self._get_layer_node_indims()
            layers = [make_layer(in_dim) for in_dim in in_dims]

        self.layers = nn.ModuleList(layers)

    @property
    def gives_edge_features(self):
        return False
