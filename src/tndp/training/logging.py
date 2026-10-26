"""Учёт бюджета, метрик и состояния цикла обучения."""
from dataclasses import dataclass, field
from typing import ClassVar
from tndp.costs.constants import METRIC_TAG_ALIASES, ACTION_PENALTIES
import time
import torch
from tndp.utils.logging import log

@dataclass
class DataCounts:
    PHASE: ClassVar[str] = "time"
    training_graphs: int | None = None
    validation_graphs: int | None = None

@dataclass
class ValidationRecord:
    PHASE: ClassVar[str] = "eval"
    cost: float
    objective: dict
    penalty: dict
    monitor: dict
    feasible_fraction: float

@dataclass
class ValueRecord:
    PHASE: ClassVar[str] = "rollout"
    value_mean: float
    value: dict

@dataclass
class DiagnosticRecord:
    PHASE: ClassVar[str] = "rollout"
    initial_ratio_error: float
    graph_instances_loaded: int
    graph_batches_loaded: int
    completed_networks: int
    return_target_mean: float
    step_reward: float
    cost: float | None = None
    ATT: float | None = field(default=None, metadata={'tag': METRIC_TAG_ALIASES['ATT']})
    RTT: float | None = field(default=None, metadata={'tag': METRIC_TAG_ALIASES['RTT']})
    feasible_fraction: float | None = None
    unserved_fraction: float | None = None
    completed_alpha_mean: float | None = None
    completed_alpha_0_fraction: float | None = None
    completed_alpha_1_fraction: float | None = None
    completed_alpha_intermediate_fraction: float | None = None

@dataclass
class ParetoRecord:
    """Preference-grid evaluation; the caller passes the prefix
    (``pareto/<mode>`` or ``eval/test/<checkpoint>/<mode>``)."""

    PHASE: ClassVar[str] = "pareto"
    ATT: float | None = field(default=None, metadata={'tag': METRIC_TAG_ALIASES['ATT']})
    RTT: float | None = field(default=None, metadata={'tag': METRIC_TAG_ALIASES['RTT']})
    feasible_fraction: float | None = None
    mean_unserved_fraction: float | None = None
    hv_penalized: float | None = None
    hv: float | None = None
    hv_reference_coverage: float | None = None
    eu_conditioned: float | None = None
    eu_archive: float | None = None
    eu_archive_feasible: float | None = None
    pareto_points_penalized: float | None = None
    pareto_points_feasible: float | None = None
    spacing_feasible: float | None = None
    sparsity_feasible: float | None = None
    distinct_networks: float | None = None
    distinct_feasible_networks: float | None = None
    distinct_pareto_networks: float | None = None
    distinct_networks_total: float | None = None
    distinct_feasible_networks_total: float | None = None
    graph_count: float | None = None
    outcome_count: float | None = None
    reference_0: float | None = None
    reference_1: float | None = None
    evaluation_seconds: float | None = None

@dataclass
class TrainRecord:
    PHASE: ClassVar[str] = "train"
    policy_gradient_loss: float
    value_loss: float
    entropy_loss: float
    clip_fraction: float
    approx_kl: float
    explained_variance: float
    diversity_loss: float
    learning_rate: float
    clip_range: float
    n_updates: int
    loss: float
    diversity_kl: float
    diversity_target: float
    diagnostics: dict = field(default_factory=dict)

@dataclass
class GoalTrainRecord:
    PHASE: ClassVar[str] = "train"
    value_loss: dict
    explained_variance: dict
    advantage_std: dict

@dataclass
class RolloutRecord:
    PHASE: ClassVar[str] = "rollout"
    ep_rew_mean: float
    ep_len_mean: float
    objective: dict
    penalty: dict

@dataclass
class TimeRecord:
    PHASE: ClassVar[str] = "time"
    fps: float
    iterations: int
    time_elapsed: float
    total_timesteps: int

def log_values(algorithm, batch, logger_list, iteration):
    scalar_values = batch.values if algorithm.scalar_values else (batch.values * batch.preferences).sum(-1)
    names = tuple(algorithm.cost.objectives)
    log(
        logger_list,
        ValueRecord(
            float(scalar_values.mean()),
            {name: float(batch.values[..., i].mean()) for i, name in enumerate(names)}
            if not algorithm.scalar_values
            else {},
        ),
        iteration,
    )

def log_iteration(env, algorithm, batch, stats, logger_list, data, budget, iteration,
                  started, episode_rewards, episode_lengths, terminal, *, learning_rate):
    scalar_rewards = batch.rewards if algorithm.scalar_values else (batch.rewards * batch.preferences).sum(-1)
    names = tuple(algorithm.cost.objectives)
    log(
        logger_list,
        TrainRecord(
            -stats.policy_objective,
            stats.value_loss,
            -stats.entropy,
            stats.clip_fraction,
            stats.approx_kl,
            stats.explained_variance,
            stats.diversity_loss,
            learning_rate,
            algorithm.params.epsilon,
            (iteration + 1) * algorithm.params.n_epochs,
            -stats.objective,
            stats.diversity_kl,
            stats.diversity_target,
            stats.diagnostics,
        ),
        iteration,
    )
    log(
        logger_list,
        GoalTrainRecord(
            dict(zip(names, stats.value_loss_by_objective)),
            dict(zip(names, stats.explained_variance_by_objective)),
            {name: float(batch.advantages[..., i].std()) for i, name in enumerate(names)}
            if batch.advantages.ndim > 2
            else {},
        ),
        iteration,
    )
    log(
        logger_list,
        RolloutRecord(
            float(episode_rewards.mean()) if episode_rewards.numel() else float("nan"),
            float(episode_lengths.float().mean()) if episode_lengths.numel() else float("nan"),
            {name: float(batch.objective_rewards[..., i].mean()) for i, name in enumerate(names)},
            {
                name: float(batch.penalty_rewards[..., i].mean())
                for i, name in enumerate((*env.result.penalty_names, *ACTION_PENALTIES))
            },
        ),
        iteration,
    )
    elapsed = time.monotonic() - started
    log(
        logger_list,
        TimeRecord(
            (iteration + 1) * budget.horizon * data.batch_size / elapsed,
            iteration + 1,
            elapsed,
            (iteration + 1) * budget.horizon * data.batch_size,
        ),
        iteration,
    )
    record = DiagnosticRecord(
        initial_ratio_error=stats.initial_ratio_error,
        graph_instances_loaded=data.graph_instances_loaded,
        graph_batches_loaded=data.graph_batches_loaded,
        completed_networks=int(batch.dones.sum()),
        return_target_mean=float(batch.returns.mean()),
        step_reward=float(scalar_rewards.mean()),
    )
    for key, values in terminal.items():
        if key != "alpha":
            setattr(record, key, float(torch.cat(values).mean()) if values else float("nan"))
    alpha = torch.cat(terminal["alpha"]) if terminal["alpha"] else None
    record.completed_alpha_mean = float(alpha.mean()) if alpha is not None else float("nan")
    for name, condition in (
        ("alpha_0", lambda a: a == 0),
        ("alpha_1", lambda a: a == 1),
        ("alpha_intermediate", lambda a: (a > 0) & (a < 1)),
    ):
        setattr(
            record,
            f"completed_{name}_fraction",
            float(condition(alpha).float().mean()) if alpha is not None else float("nan"),
        )
    log(logger_list, record, iteration)
