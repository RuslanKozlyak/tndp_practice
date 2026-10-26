"""Валидация политики и выбор checkpoint по стоимости."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import torch
from tndp.utils.logging import log
from tndp.evaluation.inference import evaluate_policy
from tndp.environment.environment import EnvParams
from tndp.costs.metrics import evaluate_network, monitor_metrics
from tndp.training.logging import ValidationRecord

@dataclass(frozen=True)
class EvaluationParams:
    validation_mode: str
    graph_count: int | None
    pareto_graph_count: int | None
    test_graph_count: int | None
    pareto_enabled: bool
    pareto_period: int
    pareto_preference_count: int
    pareto_stochastic_samples: int
    hv_reference: list
    eu_preference_count: int
    preference_count: int
    stochastic_samples: int
    seed: int
    final_test: bool

@dataclass(frozen=True)
class EvaluationProfile:
    evaluation: EvaluationParams
    env: EnvParams
    batch_size: int
    symmetric_routes: bool
    env_factory: object

class Validation:
    def __init__(self, data, cost, env, profile, device):
        self.data, self.cost, self.env = data, cost, env
        self.profile, self.device = profile, device
        self.last_monitor_step = None

    def batches(self):
        raise NotImplementedError

    @torch.no_grad()
    def run(self, policy, logger_list, step):
        collected, results = [], []
        for graphs, seeds, route_count in self.batches():
            self.env.set_route_count(route_count)
            result, routes = evaluate_policy(
                self.env,
                policy,
                graphs.to(self.device),
                self.cost.get_weights(),
                self.profile.evaluation.validation_mode,
                1,
                seed_networks=seeds,
            )
            metrics = evaluate_network(result)
            collected.append(metrics)
            results.append(result)
        metrics = {key: torch.cat([item[key] for item in collected]) for key in collected[0]}
        log(
            logger_list,
            ValidationRecord(
                cost=float(metrics["cost"].mean()),
                objective={
                    name: float(torch.cat([r.goals[:, i] for r in results]).mean())
                    for i, name in enumerate(self.cost.objectives)
                },
                penalty={
                    name: float(torch.cat([r.penalties[:, i] for r in results]).mean())
                    for i, name in enumerate(results[0].penalty_names)
                },
                monitor={name: float(value.mean()) for name, value in
                         monitor_metrics(metrics, self.cost.monitor).items()},
                feasible_fraction=float(torch.cat([~r.are_constraints_violated() for r in results]).float().mean()),
            ),
            step,
        )
        return float(metrics["cost"].mean())

    def monitor_batches(self):
        raise NotImplementedError

    def monitor(self, policy, logger_list, step, *, batches, final=False):
        """Preference fronts are evaluated by evaluation.pareto."""
        pass

class ConstructionValidation(Validation):
    def batches(self):
        for count in self.data.n_routes:
            for graphs in self.data.validation_loader:
                yield graphs, None, count

    def monitor_batches(self):
        yield self.data.monitor_graphs, None

class ValidationTracker:
    """One validation/monitor/best sequence before, after and at training end."""

    def __init__(self, validation):
        self.validation = validation
        self.best_cost, self.best_policy = float('inf'), None
        self.last_step, self.last_cost = None, None

    def _best(self, policy, cost):
        if cost < self.best_cost:
            self.best_cost, self.best_policy = cost, copy.deepcopy(policy)

    def run(self, policy, logger_list, step, *, freeze=False, final=False):
        self.last_cost = self.validation.run(policy, logger_list, step)
        self.last_step = step
        if freeze:
            policy.update_and_freeze_feature_norms()
        self.validation.monitor(policy, logger_list, step,
                                batches=self.validation.monitor_batches(), final=final)
        self._best(policy, self.last_cost)

    def finish(self, policy, logger_list, step):
        if self.last_step != step:
            self.run(policy, logger_list, step, final=True)
        else:
            self.validation.monitor(policy, logger_list, step,
                                    batches=self.validation.monitor_batches(), final=True)
            self._best(policy, self.last_cost)
