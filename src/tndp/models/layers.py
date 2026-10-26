"""Нормировки, MLP и общие операции нейросети."""
import math
from torch import nn
import logging as log
import torch

DEFAULT_NONLIN = nn.ReLU

DEFAULT_PREFERENCE_GOALS = ('ATT', 'WMC', 'RTT')

FEAT_NORM_MOMENTUM = 0.0

MLP_DEFAULT_DROPOUT = 0.0

ATTN_DEFAULT_DROPOUT = 0.0

CONST_SQRT_2 = math.sqrt(2)

CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)

CONST_INV_SQRT_2 = 1 / math.sqrt(2)

CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)

CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

class FeatureNorm(nn.Module):
    def __init__(self, dim, momentum=FEAT_NORM_MOMENTUM, unbiased=False):
        super().__init__()
        self.unbiased = unbiased
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.zeros(dim))
        self.register_buffer('avg_count', torch.tensor(1))
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('frozen', torch.tensor(False))
        self.new_mean = None
        self.new_var = None
        self.new_count = 0

    def freeze(self):
        self.frozen[...] = True

    def extra_repr(self) -> str:
        return f'mean={self.running_mean}, var={self.running_var}, ' \
            f'initialized={self.initialized}, count={self.avg_count}'

    def forward(self, xx):
        assert xx.shape[-1] == self.running_mean.shape[-1], \
            "input tensor has the wrong dimensionality!"

        x_size = xx.shape[0]
        if x_size == 0:
            # the tensor is empty, so we don't need to do anything.
            return xx

        if not self.initialized or not self.frozen:
            # just use x's own mean and variance
            old_shape = xx.shape
            xx = xx.reshape(-1, xx.shape[-1])
            x_mean = xx.mean(0).detach()
            # Legacy profiles use population variance; Holliday uses sample
            # variance. A singleton always uses the finite population fallback.
            x_var = xx.var(0, unbiased=self.unbiased and xx.shape[0] > 1).detach()
            # we're done with the need for flattened x, so reshape it
            xx = xx.reshape(old_shape)

        if not self.frozen:
            if self.new_mean is None:
                # initialize the new-samples mean
                self.new_mean = x_mean
                self.new_var = x_var
                self.new_count = x_size
            else:
                # update new-samples mean
                nmrtr = (self.new_mean * self.new_count + x_mean * x_size)
                updated_count = self.new_count + x_size
                updated_mean = nmrtr / updated_count

                # update new-samples variance using the variance-combining formula:
                # v_c = n_1(v_1 + (m_1 - m_c)^2) + n_2 * (v_2 + (m_1 - m_2)^2)
                #        / (n_1 + n_2)
                mean_diff_1 = (self.new_mean - updated_mean) ** 2
                mean_diff_2 = (x_mean - updated_mean) ** 2
                prev_part = self.new_count * (self.new_var + mean_diff_1)
                sample_part = x_size * (x_var + mean_diff_2)
                self.new_var = (prev_part + sample_part) / updated_count

                # update count of new samples
                self.new_count = updated_count

        if not self.initialized:
            # we don't have any running statistics yet, so just normalize by
             # minibatch statistics.
            shift = x_mean
            denom = x_var.sqrt()

        else:
            # avoid division by 0
            shift = self.running_mean
            denom = self.running_var.sqrt()

        # avoid division by nan or 0. nans will occur in variance if batch
         # size is 1.
        denom[(denom == 0) | denom.isnan()] = 1
        out = (xx - shift) / denom
        return out

    def update(self):
        if self.frozen:
            # do nothing
            return

        if self.new_mean is not None:
            if not self.initialized:
                # just set the initial statistics
                self.running_mean[...] = self.new_mean
                self.running_var[...] = self.new_var
                self.avg_count = torch.tensor(self.new_count)
                self.initialized[...] = True
            else:
                # update running statistics
                # scale update size in proportion to how big the sample is
                alpha = self.momentum * self.new_count / self.avg_count
                # if new count is *much* bigger than old, don't let alpha
                # be greater than 1...that would make things wierd
                alpha = min(alpha, 1.0)
                updated_mean = alpha * self.new_mean + \
                    (1 - alpha) * self.running_mean

                # implements the variance-combing formula, assuming
                 # n_1 / (n_1 + n_2) = alpha, n_2 / (n_1 + n_2) = 1 - alpha
                t1 = self.new_var + (self.new_mean - updated_mean) ** 2
                t2 = self.running_var + (self.running_mean - updated_mean) ** 2
                self.running_var[...] = alpha * t1 + (1 - alpha) * t2

                # update the running mean *after* using it to compute the new
                 # running variance
                self.running_mean[...] = updated_mean
                self.avg_count = self.momentum * self.new_count + \
                    (1 - self.momentum) * self.avg_count

            self.new_mean = None
            self.new_var = None
            self.new_count = 0

        else:
            log.warning("FeatureNorm was updated without any samples!")

def get_node_pair_descs(node_descs):
    n_nodes = node_descs.shape[-2]
    exp_dst_nodes = node_descs[:, None].expand(-1, n_nodes, -1, -1)
    exp_src_nodes = exp_dst_nodes.permute(0, 2, 1, 3)
    return torch.cat((exp_src_nodes, exp_dst_nodes), dim=-1)

def _layer_indims_helper(in_dim, embed_dim, n_layers, is_dense):
    indims = []
    if is_dense:
        curdim = in_dim
        for _ in range(n_layers):
            indims.append(curdim)
            curdim += embed_dim
    else:
        indims = [in_dim]
        indims += [embed_dim] * (n_layers - 1)
    return indims

def get_mlp(n_layers, embed_dim, nonlin_type=DEFAULT_NONLIN,
            dropout=MLP_DEFAULT_DROPOUT, in_dim=None, out_dim=None, bias=True):
    """n_layers is the number of linear layers, so the number of 'hidden
    layers' is n_layers - 1."""
    layers = []
    for li in range(n_layers):
        if li == 0 and in_dim is not None:
            layer_in_dim = in_dim
        else:
            layer_in_dim = embed_dim
        if li == n_layers - 1 and out_dim is not None:
            layer_out_dim = out_dim
        else:
            layer_out_dim = embed_dim
        layers.append(nn.Linear(layer_in_dim, layer_out_dim, bias=bias))
        if li < n_layers - 1:
            layers.append(nn.Dropout(dropout))
            layers.append(nonlin_type())
    return nn.Sequential(*layers)
