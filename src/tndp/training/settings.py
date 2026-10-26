"""Сборка модели и среды из одного профиля и параметров опыта."""
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from tndp.environment.features import FeatureParams, dims
from tndp.environment.parameters import CostConstants
from tndp.utils.checkpoints import load_model_weights
from tndp.models.encoder import GraphAttnNet
from tndp.models.policy import PathCombiningRouteGenerator
from omegaconf import OmegaConf
from tndp.costs.function import CostParams, build_cost
from tndp.costs.preferences import PreferenceSamplerParams
from tndp.environment.environment import EnvParams

@dataclass(frozen=True)
class CommonParams:
    dropout: float
    nonlin_type: str
    embed_dim: int

@dataclass(frozen=True)
class DecoderParams:
    n_nodepair_layers: int
    n_pathscorer_layers: int
    pathscorer_hidden_dim: int
    n_halt_layers: int
    n_halt_heads: int
    halt_scorer_type: str
    serial_halting: bool
    logit_clip: float | None
    feature_norm_unbiased: bool
    use_redundancy_features: bool
    temperature: float
    fixed_frequency_Hz: float
    only_routes_with_demand_are_valid: bool

@dataclass(frozen=True)
class BackboneParams:
    n_layers: int
    n_heads: int
    dense: bool
    use_norm: bool
    recurrent: bool
    residual: bool
    n_proj_layers: int
    out_dim: int | None
    layer_kwargs: dict

@dataclass(frozen=True)
class ModelParams:
    common: CommonParams
    decoder_type: str
    decoder: DecoderParams
    backbone_type: str
    backbone: BackboneParams
    features: FeatureParams

    @classmethod
    def from_values(cls, values):
        generator = values["route_generator"]
        decoder_cls = DECODER_PARAMS[generator["type"]]
        return cls(
            CommonParams(**values["common"]),
            generator["type"],
            decoder_cls(**generator["kwargs"]),
            values["backbone_gn"]["net_type"],
            BackboneParams(**values["backbone_gn"]["kwargs"]),
            FeatureParams.from_values(values["features"]),
        )

@dataclass(frozen=True)
class PolicyProfile:
    model: ModelParams
    constants: CostConstants
    objectives: tuple[str, ...]
    checkpoint_path: Path | None
    device: torch.device

POLICIES = {'PathCombiningRouteGenerator': PathCombiningRouteGenerator}

DECODER_PARAMS = {'PathCombiningRouteGenerator': DecoderParams}

BACKBONES = {'graph attn': GraphAttnNet}

def build_policy(profile: PolicyProfile):
    params = profile.model
    sizes = dims(params.features)
    backbone = BACKBONES[params.backbone_type](
        **asdict(params.common), **asdict(params.backbone), in_node_dim=sizes["node"], in_edge_dim=sizes["pair"]
    )
    policy = POLICIES[params.decoder_type](
        backbone_net=backbone,
        mean_stop_time_s=profile.constants.mean_stop_time_s,
        symmetric_routes=profile.constants.symmetric_routes,
        feature_set=params.features,
        **asdict(params.common),
        **asdict(params.decoder),
    )
    policy.objectives = profile.objectives
    policy.to(profile.device)
    if profile.checkpoint_path is not None:
        load_model_weights(policy, profile.checkpoint_path, strict=True, map_location=profile.device)
    return policy

def load_experiment(name):
    if name != 'train_construction_holliday':
        raise ValueError('Only the route construction profile is included')
    return OmegaConf.load(Path(__file__).with_name('profile.yaml'))

def cost_params(cost_cfg) -> CostParams:
    """Decode the cost group into library parameters at the run boundary."""

    return CostParams(**OmegaConf.to_container(cost_cfg.kwargs, resolve=True))

def sampler_params(preferences_cfg):
    """Decode the sampler's own preset at the run boundary."""
    values = OmegaConf.to_container(preferences_cfg, resolve=True)
    values['objectives'] = tuple(values['objectives'])
    return PreferenceSamplerParams(**values)

def policy_profile(model_cfg, cost_cfg, common, device, *, checkpoint_path=None):
    """Resolve config groups at the Run.setup boundary."""

    values = OmegaConf.to_container(cost_cfg.kwargs, resolve=True)
    return PolicyProfile(
        ModelParams.from_values(OmegaConf.to_container(model_cfg, resolve=True)),
        CostConstants(common.symmetric_routes, values["mean_stop_time_s"], values["avg_transfer_wait_time_s"]),
        tuple(values["objectives"]),
        checkpoint_path,
        device,
    )

DEVICE = torch.device('cpu')

def construction_config(cfg, rl):
    """Original Holliday profile; override budgets and the project's shared task."""
    native = load_experiment('train_construction_holliday')
    native.train_loop.n_iterations = rl.total_updates
    native.train_loop.batch_size = rl.batch_size
    native.train_loop.horizon = rl.horizon
    native.train_loop.val_period = rl.validation_every_updates
    native.algorithm.minibatch_size = rl.minibatch_size
    native.optimizer.learning_rate = rl.learning_rate
    native.algorithm.n_epochs = rl.n_epochs
    native.algorithm.gamma = rl.gamma
    native.algorithm.gae_lambda = rl.gae_lambda
    native.algorithm.epsilon = rl.clip_range
    native.algorithm.entropy_weight = rl.ent_coef
    native.algorithm.max_grad_norm = rl.max_grad_norm
    native.evaluation.validation_mode = 'greedy' if rl.validation_deterministic else 'sample'
    native.evaluation.pareto_enabled = False  # the project has its own final comparison
    native.evaluation.final_test = False
    native.cost.kwargs.demand_time_weight = cfg.alpha
    native.cost.kwargs.route_time_weight = 1 - cfg.alpha
    native.cost.kwargs.penalties.constraints = cfg.beta
    native.cost.kwargs.avg_transfer_wait_time_s = cfg.transfer_penalty * 60
    # Training samples preferences; validation and inference use cost.get_weights().
    native.preferences.variable_weights = rl.variable_cost_weights
    native.preferences.demand_time_weight = cfg.alpha
    native.preferences.route_time_weight = 1 - cfg.alpha
    return native

def components(native, cfg):
    policy = build_policy(policy_profile(native.model, native.cost, native.common, DEVICE))
    cost = build_cost(cost_params(native.cost), symmetric_routes=native.common.symmetric_routes,
                      low_memory_mode=native.cost.low_memory_mode).to(DEVICE)
    limits = EnvParams(cfg.n_routes, cfg.min_route_len, cfg.max_route_len,
                       **OmegaConf.to_container(native.env.params, resolve=True))
    return policy, cost, limits
