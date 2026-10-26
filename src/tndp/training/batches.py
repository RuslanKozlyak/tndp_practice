"""Батчи городов, веса cost и счётчики завершённых эпизодов."""
from enum import StrEnum
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from tndp.costs.preferences import PreferenceSampler, state_preferences
import random

class PreferenceDistribution(StrEnum):
    SAMPLER = "sampler"
    SIMPLEX = "simplex"

@dataclass(frozen=True)
class EpisodePreferences:
    """Training-episode preferences: distribution and lifetime. Validation keeps the configured sampler.

    ``sample`` consumes the global torch RNG and returns State weight keys -> Tensor[count].
    """
    sampler: PreferenceSampler
    distribution: PreferenceDistribution
    resample_on_terminal: bool

    def sample(self,count,device):
        if self.distribution is PreferenceDistribution.SIMPLEX:
            return state_preferences(self.sampler.sample_simplex_named(count,device))
        return state_preferences(self.sampler.sample_named(count,device))

@dataclass
class EpisodeStatistics:
    rewards: torch.Tensor
    lengths: torch.Tensor

    def record(self,rewards,ended,active):
        self.lengths -= (~active).long()
        self.rewards.add_(rewards)
        self.lengths.add_(1)
        completed_rewards=self.rewards[ended].clone()
        completed_lengths=self.lengths[ended].clone()
        self.rewards[ended]=0
        self.lengths[ended]=0
        return completed_rewards,completed_lengths

class EpisodeSource(ABC):
    def __init__(self):
        self.episode_statistics=None

    def start_episode(self,batch_size,device):
        self.episode_statistics=EpisodeStatistics(torch.zeros(batch_size,device=device),
                                                  torch.zeros(batch_size,dtype=torch.long,device=device))

    def record_step(self,rewards,ended,active):
        if self.episode_statistics is None:
            raise RuntimeError('episode counters require a started batch')
        return self.episode_statistics.record(rewards,ended,active)

    @abstractmethod
    def initialize(self): ...

    @abstractmethod
    def warmup(self,count): ...

    @abstractmethod
    def get(self,progress): ...

    @abstractmethod
    def start(self,env): ...

    @abstractmethod
    def prepare_step(self,env): ...

    @abstractmethod
    def after_step(self,env,done): ...

    @abstractmethod
    def bootstrap_mask(self,env): ...

    @abstractmethod
    def final_evaluation(self): ...

class ConstructionData(EpisodeSource):
    def __init__(self, train_loader, validation_loader, device, *, n_routes, preferences, monitor_graphs,
                 validation_indices, validation_batch_size, test_graphs, full_validation,
                 training_source_ids, validation_source_ids, test_source_ids, online):
        super().__init__()
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device
        self.n_routes = tuple(n_routes)
        self.preferences = preferences
        self.monitor_graphs = monitor_graphs
        self.validation_indices, self.validation_batch_size = validation_indices, validation_batch_size
        self.test_graphs, self.full_validation = test_graphs, full_validation
        self.training_source_ids, self.validation_source_ids, self.test_source_ids = training_source_ids, validation_source_ids, test_source_ids
        self.online = online
        self.batch_size = train_loader.batch_size
        self.generator = None
        self.finished = None
        self.route_count = None
        self.graph_instances_loaded = self.graph_batches_loaded = 0

    @property
    def training_count(self):
        return None if self.online else len(self.train_loader.dataset)

    @property
    def validation_count(self):
        return len(self.validation_loader.dataset)

    @property
    def monitor_count(self):
        return len(self.monitor_graphs)

    @property
    def final_split(self):
        return 'test' if self.test_graphs is not None else 'final_validation'

    def initialize(self):
        self.generator = iter(self.train_loader)

    def warmup(self, count):
        return ()

    def get(self, progress):
        self.route_count = (
            max(self.n_routes) if self.route_count is None else random.randint(min(self.n_routes), max(self.n_routes))
        )
        return self

    def start(self, env):
        if self.finished is None or self.finished.all():
            try:
                graphs = next(self.generator)
            except StopIteration:
                self.generator = iter(self.train_loader)
                graphs = next(self.generator)
            graphs = graphs.to(self.device)
            self.graph_instances_loaded += graphs.num_graphs
            self.graph_batches_loaded += 1
            env.set_route_count(self.route_count)
            env.reset(graphs, self.preferences.sample(graphs.num_graphs, self.device))
            self.finished = env.is_done().clone()
            self.start_episode(graphs.num_graphs,self.device)

    def prepare_step(self, env):
        pass

    def after_step(self, env, done):
        self.finished |= done
        sampled = None
        if self.preferences.resample_on_terminal and done.any():
            sampled = self.preferences.sample(int(done.sum()), self.device)
        env.reset_completed(done, sampled)

    def bootstrap_mask(self, env):
        return torch.ones(env.state.batch_size, dtype=torch.bool, device=self.device)

    def final_evaluation(self):
        graphs = self.test_graphs if self.final_split == 'test' else self.full_validation
        return graphs, None, list(range(len(graphs)))
