"""Политика construction: признаки города, оценки путей, HALT и выбор действия."""
from dataclasses import dataclass
import torch
from tndp.environment.actions import RouteAction, ActionSpec, RouteActionKind
from torch import nn
from tndp.utils.tensors import get_update_at_mask, get_variable_slice_mask
from tndp.environment.masks import (
    TORCH_FMIN,
    TORCH_FMAX,
    apply_halt_mask,
    mask_action_family,
    set_invalid_scores_to_fmin,
)
import copy
import logging as log
from tndp.data.constants import DEMAND_KEY, ROUTE_KEY
from tndp.environment.features import (
    Observation,
    aggr_edges_over_sequences,
    dims as feature_dims,
    extend_features,
    finite_time_features,
    pair_features,
)
from tndp.models.layers import (
    DEFAULT_NONLIN,
    DEFAULT_PREFERENCE_GOALS,
    MLP_DEFAULT_DROPOUT,
    FeatureNorm,
    get_mlp,
    get_node_pair_descs,
)
from tndp.models.encoder import GraphAttnNet
from tndp.models.heads import RouteEndpointsScorer, RouteScorer

@dataclass(frozen=True)
class PolicyOut:
    action: RouteAction
    logp: torch.Tensor
    entropy: torch.Tensor
    probs: torch.Tensor

    @property
    def log_probs(self):
        supported = self.probs > 0
        return torch.where(supported, self.probs, 1.0).log().masked_fill(~supported, -torch.inf)

def _joint_route_action_log_probs(scores, valid, halt_scores, serial_halting,
                                 temperature, force_halt):
    """Log probabilities of all route actions, with halt in the last column.

    Serial halting is a joint distribution: P(action) = P(continue) *
    P(action | continue). Rows without route options are deterministic halts.
    This adds no parameters and leaves the action sampler/checkpoints intact.
    """
    scores = scores.masked_fill(~valid, -torch.inf)
    no_options = ~valid.any(-1)
    forced = force_halt | no_options | (halt_scores.squeeze(-1) == TORCH_FMAX)
    # Avoid log-softmax over an all-masked row, even when it is later replaced.
    safe_scores = torch.where(no_options[:, None], torch.zeros_like(scores), scores)
    if serial_halting:
        route_log_probs = logsoftmax(safe_scores, temperature).to(torch.float64)
        decision = logsoftmax(torch.cat((-halt_scores, halt_scores), -1),
                              temperature).to(torch.float64)
        log_probs = torch.cat((route_log_probs + decision[:, :1],
                               decision[:, 1:]), -1)
    else:
        log_probs = logsoftmax(torch.cat((safe_scores, halt_scores), -1),
                               temperature).to(torch.float64)
    log_probs = log_probs.masked_fill(
        torch.cat((~valid, halt_scores <= TORCH_FMIN), -1), -torch.inf)
    deterministic = torch.full_like(log_probs, -torch.inf)
    deterministic[:, -1] = 0.
    return torch.where(forced[:, None], deterministic, log_probs)

def select(scores, scores_as_logits=False, softmax_temp=1,
           n_selections=1, selection=None):
    """scores: an n_options or batch_size x n_options tensor."""
    if scores.ndim == 1:
        # add a batch dimension
        scores = scores[None, :]
    logits = logsoftmax(scores, softmax_temp).to(dtype=torch.float64)

    # do this to avoid nans from 0 * -inf
    ent_logits = get_update_at_mask(logits, logits.isinf(), 0)
    entropy = -(ent_logits.exp() * ent_logits).sum(dim=-1)
    assert not entropy.isnan().any()

    if selection is None:
        if not scores_as_logits:
            _, selection = logits.topk(n_selections)
        else:
            probs = logits.exp()
            selection = probs.multinomial(n_selections)
            # if fewer than n_selections options survive rounding down by exp()
             # then use topk to select the top n_selections
            needs_topk = (probs > 0).sum(dim=1) < n_selections
            if needs_topk.any():
                _, topk_selection = logits[needs_topk].topk(n_selections)
                selection[needs_topk] = topk_selection

        selection.squeeze_(-1)

    if n_selections == 1:
        # add a "selection" dimension
        selection = selection[..., None]
        if scores.ndim == 1:
            selected_logits = logits[selection]
        else:
            # there is a batch dimension
            selected_logits = logits.gather(1, selection)

    else:
        # renormalize the log-probabilities after each selection
        idxs = torch.arange(logits.shape[1]).expand(logits.shape[0], -1)
        idxs = idxs.to(scores.device)
        selection_mask = idxs[..., None] == selection[:, None]
        selection_mask = selection_mask.any(dim=-1)
        selection_neutralizer = torch.zeros_like(scores)
        selection_neutralizer[selection_mask] = TORCH_FMIN
        unselected_scores = scores + selection_neutralizer
        if scores.ndim == 1:
            selected_scores = scores[selection]
        else:
            # there is a batch dimension
            selected_scores = scores.gather(1, selection)

        # flip the selected scores so the first selected comes last
        reordered_scores = \
            torch.cat((unselected_scores, selected_scores.flip(-1)), dim=-1)
        # sum them so the last element is all scores, 2nd last is all but
         # first picked, etc.
        logsumexps = torch.logcumsumexp(reordered_scores, -1)
        # flip again so the first selected comes first
        logdenoms = logsumexps[..., -n_selections:].flip(-1)
        selected_logits = selected_scores - logdenoms

    return selection, selected_logits, entropy

def logsoftmax(energies, softmax_temp=1):
    return nn.functional.log_softmax(energies / softmax_temp, dim=-1)

class RouteGeneratorBase(nn.Module):
    # Feature spec: order of the cost-weight slots in every positional
    # feature vector this model consumes (global state features, edge
    # planes, decoder context). Locked to the trained checkpoints (see
    # DEFAULT_PREFERENCE_GOALS); a model trained with a different order overrides
    # this next to its checkpoint.
    preference_goals = DEFAULT_PREFERENCE_GOALS

    def __init__(self, backbone_net, mean_stop_time_s,
                 embed_dim, n_nodepair_layers, nonlin_type=DEFAULT_NONLIN,
                 dropout=MLP_DEFAULT_DROPOUT, symmetric_routes=True,
                 temperature=1, fixed_frequency_Hz=0.00074,
                 only_routes_with_demand_are_valid=False, feature_set=None):
        """nonlin_type: a nonlin, or the string name of a nonlin, like 'ReLU'
            or 'LeakyReLU' or 'Tanh'"""
        super().__init__()
        if feature_set is None:
            raise ValueError("policy requires a feature set")
        self.feature_set = feature_set
        self.preference_goals = feature_set.preference_goals
        # make this a parameter of the model so it will be kept in state_dict
        # mst = torch.tensor(mean_stop_time_s, dtype=torch.float32)
        self.mean_stop_time_s = mean_stop_time_s

        self.dropout = dropout
        if isinstance(nonlin_type, str):
            self.nonlin_type = getattr(nn, nonlin_type)
        else:
            self.nonlin_type = nonlin_type
        self.embed_dim = embed_dim
        self.n_nodepair_layers = n_nodepair_layers
        self.backbone_net = backbone_net
        self.symmetric_routes = symmetric_routes
        self.temperature = temperature
        self.fixed_freq = fixed_frequency_Hz
        self.only_routes_with_demand_are_valid = \
            only_routes_with_demand_are_valid

        self.edge_feat_dim = backbone_net.in_edge_dim
        if self.edge_feat_dim is None:
            self.edge_feat_dim = 0
        # + 2 for the cost weights, + 2 for the two n_routes features
        if symmetric_routes:
            self.full_nodepair_dim = self.edge_feat_dim + embed_dim
        else:
            self.full_nodepair_dim = self.edge_feat_dim + embed_dim * 2

        # edge feats: demand, drive time, has route, route time, has street,
         # street time, has 1-transfer path, has 2-transfer path, has any path,
         # shortest transit path time, is same node
        if self.edge_feat_dim > 0:
            self.edge_norm = FeatureNorm(self.edge_feat_dim - 1)
            self.time_norm = FeatureNorm(1)
        self.node_norm = FeatureNorm(backbone_net.in_node_dim)

    @property
    def edge_key_order(self):
        return (DEMAND_KEY, ROUTE_KEY)

    def update_and_freeze_feature_norms(self):
        self.update_feature_norms()
        self.freeze_feature_norms()

    def update_feature_norms(self):
        for mod in self.modules():
            if isinstance(mod, FeatureNorm):
                mod.update()

    def freeze_feature_norms(self):
        for mod in self.modules():
            if isinstance(mod, FeatureNorm):
                mod.freeze()



    def _get_edge_features(self, state):
        raw = pair_features(state, use_redundancy=self.use_redundancy_features,
                            preference_goals=self.preference_goals)
        edge_features = self.edge_norm(raw)
        finite_drive_times = finite_time_features(state, state.drive_times)
        drive_times = self.time_norm(finite_drive_times[..., None])
        return torch.cat((edge_features, drive_times), dim=-1)

    @staticmethod
    def fold_node_descs(node_descs, state):
        # fold the node description vectors so that the batch dimension
         # is separate from the nodes-in-a-batch-elem dimension
        folded_node_descs = torch.zeros((state.batch_size, state.max_n_nodes,
                                         node_descs.shape[-1]),
                                         device=node_descs.device)
        node_pad_mask = torch.zeros((state.batch_size, state.max_n_nodes),
                                    dtype=bool, device=node_descs.device)
        node_offset = 0
        for bi, num_nodes in enumerate(state.n_nodes):
            num_nodes = int(num_nodes.item())
            next_offset = node_offset + num_nodes
            folded_node_descs[bi, :num_nodes, :node_descs.shape[-1]] = \
                node_descs[node_offset:next_offset]
            node_pad_mask[bi, num_nodes:] = True
            node_offset = next_offset
        if node_offset != node_descs.shape[0]:
            raise ValueError(
                "Node embedding count does not match batched graph nodes: "
                f"{node_offset} vs {node_descs.shape[0]}"
            )

        return folded_node_descs, node_pad_mask

class PathCombiningRouteGenerator(RouteGeneratorBase):
    action_spec = ActionSpec("construction")
    def __init__(self, *args, n_pathscorer_layers=3, pathscorer_hidden_dim=16,
                 halt_scorer_type='endpoints', n_halt_layers=3, n_halt_heads=4,
                 logit_clip=None, serial_halting=True,
                 use_redundancy_features=False, feature_norm_unbiased=False,
                 halt_scorer: RouteScorer | None = None, **kwargs):
        """Generates routes by combining shortest paths.

        n_pathscorer_layers -- number of layers in the MLP that updates path
            scores
        pathscorer_hidden_dim -- number of hidden units in the MLP that updates
            path scores
        halt_scorer_type -- 'endpoints' for the built-in scorer.
        halt_scorer -- an optional, explicitly constructed RouteScorer.
            It receives the same node embeddings and configured features.
        n_halt_layers -- number of layers in the network that computes the
            halting score.
        n_halt_heads -- number of heads in the network that computes the
            halting score, if it has heads.
        logit_clip -- if not None, then the logits of path extensions are
            clipped to this value.
        serial_halting -- if True, the halt score is used to make a binary
            halt-or-continue choice before choosing each extension.  If False,
            the 'halt' action is part of the same set of actions as the
            extensions.
        """
        super().__init__(*args, **kwargs)
        self.logit_clip = logit_clip
        self.serial_halting = serial_halting
        self.use_redundancy_features = use_redundancy_features
        self.n_redundancy_global_feats = 5 if use_redundancy_features else 0
        self.n_redundancy_path_feats = 5 if use_redundancy_features else 0
        if use_redundancy_features and self.edge_feat_dim != 18:
            raise ValueError(
                "Redundancy-aware path combining expects a backbone with "
                f"in_edge_dim=18, got {self.edge_feat_dim}")
        self.nodepair_scorer = get_mlp(self.n_nodepair_layers,
                                       self.embed_dim,
                                       self.nonlin_type, self.dropout,
                                       in_dim=self.full_nodepair_dim,
                                       out_dim=1)

        assert self.only_routes_with_demand_are_valid is False, 'not supported'
        self._route_step_counts = []

        # Gated adjustment conditioning: when > 0, get_global_state_features
        # appends this many extra features (target + normalized weight), so the
        # scorers that consume global features must widen by the same amount.
        # Default 0 keeps the construction-model dims (and frozen weights) intact.
        self.n_adjustment_cond_feats = len(self.feature_set.adjustment_features)
        _halt_n_extra = feature_dims(self.feature_set)["halt"]

        path_scorer_indim = feature_dims(self.feature_set)["extend"]
        # self.path_input_norm = FeatureNorm(FEAT_NORM_MOMENTUM,
        #                                    path_scorer_indim - 1)
        self.path_scorer = nn.Sequential(
            FeatureNorm(path_scorer_indim),
            get_mlp(n_pathscorer_layers, pathscorer_hidden_dim,
                    self.nonlin_type, self.dropout, in_dim=path_scorer_indim,
                    out_dim=1)
        )
        if halt_scorer is not None:
            if not isinstance(halt_scorer, RouteScorer):
                raise TypeError("halt_scorer must implement RouteScorer")
            self.halt_scorer = halt_scorer
        elif halt_scorer_type == "endpoints":
            self.halt_scorer = RouteEndpointsScorer(self.embed_dim,
                                                    self.nonlin_type,
                                                    self.dropout,
                                                    n_halt_layers,
                                                    n_extra_feats=_halt_n_extra,
                                                    preference_goals=self.preference_goals)
        else:
            raise ValueError(
                "The default halt scorer is 'endpoints'; pass an explicit "
                "halt_scorer from policy.models_additional for another scorer"
            )

        # The upstream construction profile uses sample variance. Preserve
        # legacy checkpoint profiles' population variance by default.
        for module in self.modules():
            if isinstance(module, FeatureNorm):
                module.unbiased = feature_norm_unbiased

        # self.value_net = ValueNetScorer(self.embed_dim, self.nonlin_type,
        #                                 self.dropout, n_halt_layers)

    def clear_step_counts_log(self):
        """Clear the route step counts accumulated by rollout."""
        self._route_step_counts = []

    @property
    def route_step_counts(self):
        """Owned rollout statistics exposed as an immutable sequence."""
        return tuple(self._route_step_counts)

    def record_step_counts(self, counts: list):
        self._route_step_counts.extend(counts)

    def act(self, obs, mask, greedy=False):
        if not isinstance(obs, Observation):
            raise TypeError("policy execution requires an Observation")
        return self._action_output(obs, mask, greedy, None)

    def forward(self, obs, mask, greedy=False):
        return self.act(obs, mask, greedy)

    def evaluate(self, obs, mask, action):
        if not isinstance(obs, Observation):
            raise TypeError("policy execution requires an Observation")
        return self._action_output(obs, mask, False, action)

    def _action_output(self, obs, mask, greedy, action):
        indices, logp, entropy, log_probs = self._decode(
            obs, greedy, None if action is None else action.indices,
            obs.encoding, mask=mask)
        kind = torch.where(indices[:, 0] < 0, RouteActionKind.HALT, RouteActionKind.EXTEND)
        return PolicyOut(RouteAction(kind, indices), logp, entropy, log_probs.exp())


    def _decode(self, state, greedy=False, actions=None,
             precalc_data=None, *, mask):
        """Take an action for the given state.
        actions -- a batch_size x 2 tensor of predetermined actions to take. If
            None, the actions will be chosen by the model.  If the value is
            (-1, -1), that means to halt.  Otherwise, the first and second
            columns give the starting and ending nodes of the path segment to
            add.
        """
        log.debug("stepping")
        if not isinstance(state, Observation):
            raise TypeError("policy execution requires an Observation")

        if precalc_data is None:
            node_descs, node_pad_mask, np_embeds, path_scores = \
                self.encode(state)
        else:
            node_descs, node_pad_mask, np_embeds, path_scores = precalc_data

        path_seqs = state.paths
        batch_size = state.batch_size

        # don't choose segments greater than the max route length
        candidates = mask.require_candidates()
        path_lens = (path_seqs > -1).sum(dim=-1)
        current_route_lens = state.current_route_n_stops
        paths_are_invalid = ~candidates.path_valid
        path_scores = set_invalid_scores_to_fmin(path_scores,
                                                      ~paths_are_invalid)

        # determine if we're extending a route or starting a new one
        # for starting, we ignore the halt choice
        starting = ~state.done & (current_route_lens == 0)

        # update scores and get validity for extensions to non-empty routes
        padding = (0,1, 0,1, 0,0)
        drive_times = torch.nn.functional.pad(state.drive_times, padding)
        padding = (0, 0) + padding
        np_embeds = torch.nn.functional.pad(np_embeds, padding)
        getext_args = (state, current_route_lens, path_seqs,
                       np_embeds, drive_times, path_scores, path_lens)
        prev_scores, prev_valid = self._get_extension_scores(*getext_args,
            before_or_after='before', valid=candidates.before_valid)
        next_scores, next_valid = self._get_extension_scores(*getext_args,
            before_or_after='after', valid=candidates.after_valid)

        # compute the scores for halting
        halt_scores = self.halt_scorer(state, node_descs, state.current_routes,
                                       state.current_route_time, node_pad_mask)
        ext_valid = prev_valid | next_valid
        halt_scores = apply_halt_mask(halt_scores, candidates)
        old_route_is_done = candidates.old_route_is_done

        ext_scores = prev_scores * prev_valid + next_scores * next_valid + \
            TORCH_FMIN * ~ext_valid
        start_scores = self._update_path_scores(state, path_scores, path_lens,
                                                state.drive_times,
                                                candidate_paths=path_seqs)
        # The conditional continue branch must not depend on the replayed
        # action: a sampled halt still needs every extension in its KL.
        continuing = (~starting & ~old_route_is_done)[:, None, None]
        all_scores = torch.where(continuing, ext_scores, start_scores)
        all_valid = candidates.extend_valid
        all_scores, all_valid = mask_action_family(all_scores, all_valid, mask.extend)
        action_log_probs = _joint_route_action_log_probs(
            all_scores.flatten(1), all_valid.flatten(1), halt_scores,
            self.serial_halting, self.temperature, old_route_is_done)

        route_is_done = old_route_is_done
        if self.serial_halting:
            # decide whether to halt
            continue_scores = -halt_scores
            cont_or_halt = torch.cat((continue_scores, halt_scores), dim=-1)

            given_halt_actions = None
            if actions is not None:
                # predetermined actions were given, so force them to be chosen
                given_halt_actions = (actions[:, 0] == -1).to(torch.long)

            halt, corh_logit, corh_ent = select(cont_or_halt, not greedy,
                                                self.temperature,
                                                selection=given_halt_actions)
            assert (corh_logit > TORCH_FMIN).all(), "halt score is too low!"
            corh_logit = corh_logit.squeeze(1)
            chose_halt = halt.squeeze(1).bool()

            # update doneness
            route_is_done = old_route_is_done | chose_halt

        # for extending, we care whether we chose to halt
        extending = ~starting & ~route_is_done
        xtnding_exp = extending[:, None, None]
        ppe_scores = start_scores * ~xtnding_exp + ext_scores * xtnding_exp
        ppe_scores, _ = mask_action_family(ppe_scores, all_valid, mask.extend)

        # select an option
        flat_path_scores = ppe_scores.reshape(batch_size, -1)
        if not self.serial_halting:
            flat_path_scores = torch.cat((flat_path_scores, halt_scores),
                                         dim=-1)
            halt_idx = flat_path_scores.shape[-1] - 1

        given_ext_actions = None
        if actions is not None:
            # predetermined actions were given, so force them to be chosen
            given_ext_actions = actions[:, 0] * state.n_nodes + actions[:, 1]
            if self.serial_halting:
                given_ext_actions[route_is_done] = 0
            else:
                given_ext_actions[route_is_done] = halt_idx
                given_ext_actions[actions[:, 0] < 0] = halt_idx

        flat_idxs, ext_logit, ext_ent = select(flat_path_scores, not greedy,
                                               self.temperature,
                                               selection=given_ext_actions)

        flat_idxs = flat_idxs.squeeze(-1)
        ext_logit = ext_logit.squeeze(-1)
        if self.serial_halting:
            # these don't contribute if we chose to halt
            ext_logit[chose_halt] = 0.0
            ext_ent[chose_halt] = 0.0
        else:
            # mark the halt action as chosen
            chose_halt = flat_idxs == halt_idx
            route_is_done = route_is_done | chose_halt

        assert ((ext_logit > TORCH_FMIN) | route_is_done).all()

        from_idxs = torch.div(flat_idxs, state.n_nodes, rounding_mode='floor')
        to_idxs = flat_idxs % state.n_nodes
        folded_idxs = torch.stack((from_idxs, to_idxs), dim=-1)
        set_to_minus1 = route_is_done[:, None]
        folded_idxs = folded_idxs * ~set_to_minus1 + -1 * set_to_minus1

        # combine the logits and entropies of the decisions
        if self.serial_halting:
            logits = corh_logit + ext_logit
            entropy = corh_ent + ext_ent
        else:
            logits = ext_logit
            entropy = ext_ent

        # is_halt = (folded_idxs == -1).any(-1)
        # badlen = (current_route_lens < state.min_route_len) & \
        #     (current_route_lens > 0)
        # if (is_halt & badlen).any():
        #     log.warning("Halting a route that's too short!")

        return folded_idxs, logits, entropy, action_log_probs

    def encode(self, state: Observation):
        """Encode an Observation for actor decisions and the shared critic."""
        if not isinstance(state, Observation):
            raise TypeError("policy execution requires an Observation")
        # assemble route data objects
        input_batch = copy.copy(state.graph_data)
        input_batch.x = state.norm_node_features
        input_batch.edge_index = input_batch[DEMAND_KEY].edge_index
        edge_feats_square = self._get_edge_features(state)
        if isinstance(self.backbone_net, GraphAttnNet) or self.backbone_net.gives_edge_features:
            isedge_mask = torch.eye(state.max_n_nodes, device=state.device)
            isedge_mask = ~(isedge_mask.bool())
            if state.max_n_nodes * state.batch_size > input_batch.num_nodes:
                # some graphs are smaller than the max size, so mask out the
                 # invalid node locations.
                isedge_mask = isedge_mask.repeat(state.batch_size, 1, 1)
                for bi in range(state.batch_size):
                    isedge_mask[bi, state.n_nodes[bi]:] = False
                    isedge_mask[bi, :, :state.n_nodes[bi]:] = False
            else:
                isedge_mask = isedge_mask.expand(state.batch_size, -1, -1)

            edge_feats_flat = edge_feats_square[isedge_mask]
            input_batch.edge_attr = edge_feats_flat

        path_seqs = state.paths

        # run GNN forward
        if self.backbone_net.gives_edge_features:
            node_descs, _ = self.backbone_net(input_batch)
        else:
            node_descs = self.backbone_net(input_batch)

        node_descs, node_pad_mask = self.fold_node_descs(node_descs, state)
        if self.symmetric_routes:
            # just sum them
            np_embeds = node_descs[:, None] + node_descs[:, :, None]
        else:
            # concatenate them along the last dimension
            np_embeds = get_node_pair_descs(node_descs)
        np_embeds = torch.cat((np_embeds, edge_feats_square), dim=-1)
        np_scores = self.nodepair_scorer(np_embeds).squeeze(-1)
        score_clip = 10**6
        if (not torch.isfinite(np_scores).all()) or \
                (np_scores.abs() >= score_clip).any():
            log.warning(
                "Nodepair scores exceeded the safe range; clipping for "
                "route generation.")
            np_scores = torch.nan_to_num(
                np_scores, nan=0.0, posinf=score_clip, neginf=-score_clip)
            np_scores = np_scores.clamp(-score_clip, score_clip)

        path_scores = aggr_edges_over_sequences(path_seqs,
                                                   np_scores[..., None], 'sum')
        path_scores.squeeze_(-1)

        return node_descs, node_pad_mask, np_embeds, path_scores

    def get_critic_features(self, state: Observation):
        """Pooled GNN node embeddings for a shared-encoder critic.

        Returns ``[batch, 2 * embed_dim + n_global_features]``:
        graph mean-pool || current-route mean-pool || global state feats.

        The encoder forward is detached: value-function gradients do not
        flow back into the actor's GNN. This gives the critic the same
        rich state representation the actor sees, without destabilizing
        the policy via mixed value/policy gradients.
        """
        if not isinstance(state, Observation):
            raise TypeError("policy execution requires an Observation")
        with torch.no_grad():
            node_descs, node_pad_mask, _, _ = self.encode(state)
        node_descs = node_descs.detach()

        # Graph-level mean pool over valid (non-padding) nodes.
        valid = (~node_pad_mask).to(node_descs.dtype).unsqueeze(-1)
        graph_embed = (node_descs * valid).sum(dim=1) / \
            valid.sum(dim=1).clamp_min(1.0)

        # Pool the embeddings of the nodes on the route currently being
        # edited so the critic sees the specific route, not just the graph.
        routes = state.current_routes
        route_mask = (routes >= 0)
        safe_routes = routes.clamp_min(0)
        gather_idx = safe_routes.unsqueeze(-1).expand(
            -1, -1, node_descs.shape[-1])
        route_node_descs = torch.gather(node_descs, 1, gather_idx)
        route_valid = route_mask.to(node_descs.dtype).unsqueeze(-1)
        route_embed = (route_node_descs * route_valid).sum(dim=1) / \
            route_valid.sum(dim=1).clamp_min(1.0)

        global_feats = state.global_features.to(node_descs.dtype)
        return torch.cat([graph_embed, route_embed, global_feats], dim=-1)

    def _get_extension_scores(self, state: Observation, route_lens,
                              all_paths, nodepair_embeds, drive_times,
                              path_scores, path_lens, before_or_after='after', *, valid):
        log.debug("get extension scores")
        assert before_or_after in ['before', 'after']
        lens_wo_term = route_lens - 1
        # clamp for cases where the route is empty, so we don't get -1
        lens_wo_term.clamp_(min=0)
        times_from_start = state.current_route_times_from_start
        routes = state.current_routes
        final_times = times_from_start.gather(1, lens_wo_term[:, None])
        times_to_end = (times_from_start - final_times).abs()
        if before_or_after == 'after':
            # terminal is last node
            mask = get_variable_slice_mask(routes, 1, froms=lens_wo_term)
            routes_but_terminal = get_update_at_mask(routes, mask, -1)[:, :-1]
            terminal_nodes = routes.gather(1, lens_wo_term[:, None]).squeeze(-1)
            times_to_end = get_update_at_mask(times_to_end, mask, 0)[:, :-1]
        else:
            # terminal is first node
            routes_but_terminal = routes[:, 1:].clone()
            terminal_nodes = routes[:, 0]
            times_to_end = times_to_end[:, 1:]
            # swap from- and to-node axes so indexing the first node axis
             # by the terminal gives data on edges/paths *to* the terminal,
             # instead of *from* the terminal.
            drive_times = drive_times.transpose(-2, -1)
            nodepair_embeds = nodepair_embeds.transpose(-3, -2)
            path_scores = path_scores.transpose(-2, -1)
            path_lens = path_lens.transpose(-2, -1)
            all_paths = all_paths.transpose(-3, -2)

        # get possible extensions
        batch_idxs = state.batch_indices
        raw_extension_paths = all_paths[batch_idxs, terminal_nodes]
        extension_paths = raw_extension_paths.clone()
        # ignore the terminal itself in the extensions
        extension_paths[extension_paths == terminal_nodes[:, None, None]] = -1

        times_from_end = drive_times[batch_idxs, terminal_nodes]
        # batch_size x route_len - 1 x n_nodes + 1
        times_on_route = times_from_end[:, None] + times_to_end[:, :, None]

        # compute scores for all edges from each node on route
        embeds_on_route = nodepair_embeds[batch_idxs[:, None],
                                          routes_but_terminal]

        # use a neural network to compute scores and lengths
        times_on_route = finite_time_features(state, times_on_route)
        norm_times = self.time_norm(times_on_route.flatten(0, 2)[:, None])
        norm_times = norm_times.reshape_as(times_on_route)[..., None]
        log.debug("updating edge scores...")
        embeds_w_newtimes = torch.cat((embeds_on_route[..., :-1], norm_times),
                                      dim=-1)
        from_route_edge_scores = self.nodepair_scorer(embeds_w_newtimes)
        from_route_edge_scores = from_route_edge_scores.squeeze(-1)
        log.debug("edge scores updated")

        # zero out scores corresponding to dummy route nodes
        from_route_edge_scores = \
            from_route_edge_scores * (routes_but_terminal > -1)[..., None]
        # sum over nodes on route to get "score" of visiting each node before/
         # after the current route.
        from_route_node_scores = from_route_edge_scores.sum(dim=-2)
        # compute scores for each possible following sequence
        # ignore first node of each sequence, since it's the terminal
        from_route_pathnode_scores = \
            from_route_node_scores[batch_idxs[:, None, None], extension_paths]
        # zero locations corresponding to "dummy" extension path nodes
        from_route_pathnode_scores[extension_paths == -1] = 0
        from_route_path_scores = from_route_pathnode_scores.sum(dim=-1)

        # add score of route from end node to each node not on route, and add
         # current score
        innate_scores = path_scores[batch_idxs, terminal_nodes]
        ext_scores = from_route_path_scores + innate_scores

        # apply final transformer of lengths
        new_path_lens = path_lens[batch_idxs, terminal_nodes]
        # max over the stops already on the route, and drop the dummy row
        added_drive_times = times_from_end[..., :-1]

        ext_scores = self._update_path_scores(state, ext_scores, new_path_lens,
                                              added_drive_times, route_lens,
                                              times_to_end.max(-1)[0],
                                              raw_extension_paths)

        assert ext_scores.isfinite().all()

        out_scores = torch.zeros_like(path_scores)
        out_scores[batch_idxs, terminal_nodes] = ext_scores

        if before_or_after == 'before':
            # swap from- and to-node axes back
            out_scores = out_scores.transpose(-2, -1)
        out_valids = valid
        out_scores = get_update_at_mask(out_scores, ~out_valids, TORCH_FMIN)

        log.debug("extension scores gotten")

        return out_scores, out_valids

    # def _update_path_scores(self, state, base_path_scores, new_path_lens,
    #                         new_drive_times):
    #     """Update the path scores based on the new path lengths and drive times.
    #     """
    #     if (base_path_scores > 10**5).any():
    #         log.warning("base path scores are blowing up!")

    #     is_valid = base_path_scores > TORCH_FMIN
    #     update_in = torch.stack((base_path_scores, new_drive_times,
    #                              new_path_lens), dim=-1)

    #     global_feat = global_state_features(state, )
    #     update_in = nn.functional.pad(update_in, (0, global_feat.shape[-1]))
    #     for _ in range(update_in.ndim - global_feat.ndim):
    #         global_feat = global_feat.unsqueeze(1)
    #     update_in[..., -global_feat.shape[-1]:] = global_feat
    #     update_in = update_in[is_valid]

    #     updated_path_scores_wo_infs = self.path_scorer(update_in).squeeze(-1)
    #     if self.logit_clip is not None:
    #         upswi = torch.tanh(updated_path_scores_wo_infs)
    #         updated_path_scores_wo_infs = torch.clamp(upswi, -self.logit_clip,
    #                                                   self.logit_clip)
    #     updated_path_scores = torch.full_like(base_path_scores, TORCH_FMIN)
    #     updated_path_scores[is_valid] = updated_path_scores_wo_infs

    #     return updated_path_scores

    def _update_path_scores(self, state: Observation, base_path_scores,
                            new_path_lens, new_drive_times, prev_len=None,
                            prev_drive_time=None, candidate_paths=None):
        """Update the path scores based on the new path lengths and drive times.
        """
        is_valid = base_path_scores > TORCH_FMIN
        update_in = extend_features(state, base_path_scores, new_path_lens,
                                    new_drive_times, prev_len, prev_drive_time,
                                    candidate_paths)
        update_in = update_in[is_valid]

        updated_path_scores_wo_infs = self.path_scorer(update_in).squeeze(-1)
        if self.logit_clip is not None:
            upswi = torch.tanh(updated_path_scores_wo_infs)
            updated_path_scores_wo_infs = torch.clamp(upswi, -self.logit_clip,
                                                      self.logit_clip)
        updated_path_scores = torch.full_like(base_path_scores, TORCH_FMIN)
        updated_path_scores[is_valid] = updated_path_scores_wo_infs

        return updated_path_scores
