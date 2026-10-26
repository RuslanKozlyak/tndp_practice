"""config"""
from dataclasses import dataclass, field

DEFAULT_METRICS = ('cost', 'ATT', 'RTT', 'd0', 'd1', 'd2', 'un')

@dataclass(frozen=True)
class ProblemConfig:
    n_nodes: int = 15
    n_routes: int = 3
    min_route_len: int = 2
    max_route_len: int = 8
    alpha: float = 0.8
    beta: float = 5.0
    transfer_penalty: float = 5.0  # all physical times are minutes
    seed: int = 42

    def __post_init__(self):
        if not 2 <= self.min_route_len <= self.max_route_len <= self.n_nodes:
            raise ValueError('Require 2 <= MIN <= MAX <= n_nodes')
        if self.n_routes < 1 or not 0 <= self.alpha <= 1:
            raise ValueError('Require n_routes > 0 and alpha in [0, 1]')
        if self.beta < 0 or self.transfer_penalty < 0:
            raise ValueError('Penalties must be nonnegative')

@dataclass(frozen=True)
class GAConfig:
    population_size: int = 16
    generations: int = 10
    mutation_steps: int = 10  # E: mutation rounds before recruitment
    shorten_probability: float = 0.2
    mutation_probability: float = 1.0

    def __post_init__(self):
        if self.population_size < 2 or self.population_size % 2 or min(self.generations, self.mutation_steps) < 1:
            raise ValueError('Require an even population >= 2, generations >= 1 and mutation_steps >= 1')
        if any(not 0 <= p <= 1 for p in (self.mutation_probability, self.shorten_probability)):
            raise ValueError('Probabilities must be in [0, 1]')

@dataclass(frozen=True)
class PPOConfig:
    total_updates: int = 10
    learning_rate: float = 1.6e-3
    batch_size: int = 32  # cities in a collection batch
    horizon: int = 120  # actions collected per city
    minibatch_size: int = 128  # transitions per optimizer step
    n_epochs: int = 1
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    max_grad_norm: float = 0.5
    ent_coef: float = 0.0
    validation_deterministic: bool = False
    variable_cost_weights: bool = True
    validation_every_updates: int = 10

    def __post_init__(self):
        if min(self.total_updates, self.horizon, self.batch_size, self.minibatch_size,
               self.n_epochs, self.validation_every_updates) < 1:
            raise ValueError('PPO sizes must be positive')
        if self.minibatch_size < 2 or self.minibatch_size % self.batch_size or self.rollout_size % self.minibatch_size:
            raise ValueError('minibatch_size >= 2 must contain full city batches and divide batch_size * horizon')
        if self.learning_rate <= 0 or self.ent_coef < 0 or not 0 <= self.gamma <= 1:
            raise ValueError('Invalid PPO parameters')
        if not 0 <= self.gae_lambda <= 1 or not 0 < self.clip_range < 1:
            raise ValueError('Require gae_lambda in [0, 1] and clip_range in (0, 1)')
        if self.max_grad_norm <= 0:
            raise ValueError('max_grad_norm must be positive')

    @property
    def rollout_size(self):
        return self.horizon * self.batch_size

    @property
    def total_timesteps(self):
        """Transition budget derived from complete PPO updates."""
        return self.total_updates * self.rollout_size

    @classmethod
    def from_checkpoint(cls, values):
        """Read native checkpoints saved before parameter names were clarified."""
        from dataclasses import fields
        values = dict(values)
        if 'n_envs' in values:
            values['minibatch_size'] = values['batch_size']
            values['batch_size'] = values['n_envs']
            values['horizon'] = values['n_steps']
            size = values['batch_size'] * values['horizon']
            values['validation_every_updates'] = max(1, (values['eval_every'] + size - 1) // size)
        names = {field.name for field in fields(cls)}
        return cls(**{name: value for name, value in values.items() if name in names})

@dataclass(frozen=True)
class DatasetConfig:
    train_size: int = 900
    val_size: int = 100
    test_size: int = 4
    source: str | None = None  # generate cities; optionally load a saved datasets.npz
    graph_type: str = 'mixed'
    edge_keep_probability: float = 0.7
    side_km: float = 30.0
    speed_kmh: float = 54.0

    def __post_init__(self):
        if min(self.train_size, self.val_size, self.test_size) < 1:
            raise ValueError('Dataset sizes must be positive')

@dataclass(frozen=True)
class BenchmarkConfig:
    path: str = 'benchmark/mumford0'
    n_nodes: int = 30
    n_routes: int = 12
    min_route_len: int = 2
    max_route_len: int = 15

@dataclass(frozen=True)
class ExperimentConfig:
    """Construction policy, its PPO and held-out evaluation settings."""
    output_dir: str = 'artifacts/results/experiment_01'
    train: bool = True
    pretrained_weights: str | None = None  # standalone GAT weights; skip training
    cpu_threads: int = 1
    problem: ProblemConfig = field(default_factory=lambda: ProblemConfig(
        n_nodes=20, n_routes=10, min_route_len=2, max_route_len=12))
    data: DatasetConfig = field(default_factory=DatasetConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    ea: GAConfig = field(default_factory=lambda: GAConfig(
        population_size=12, generations=10, mutation_steps=2))
    lc_samples: int = 100
    pareto_weights: tuple[float, ...] = tuple(i / 10 for i in range(11))
    metrics: tuple[str, ...] = DEFAULT_METRICS
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def __post_init__(self):
        names = tuple('cost' if name == 'C' else name for name in self.metrics)
        if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError('metrics must contain distinct nonempty column names')
        object.__setattr__(self, 'metrics', names)
        if self.lc_samples < 2:
            raise ValueError('Require lc_samples >= 2')
        if not self.pareto_weights or any(not 0 <= a <= 1 for a in self.pareto_weights):
            raise ValueError('Pareto weights must be in [0, 1]')
