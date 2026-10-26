"""Пользовательская функция стоимости для пакетной среды и сохранения опыта."""
from dataclasses import dataclass, field, replace
from math import isfinite
from types import MappingProxyType
from pathlib import Path
import cloudpickle
import torch
from torch import nn
from tndp.costs.objective import network_cost
from tndp.costs.state import accepts_state, call_cost, native_cost_states

@dataclass(frozen=True, eq=False)
class CostFunction:
    """Compatibility with previously saved cost_function.pkl objects.

    New notebook examples pass Python functions directly.
    """
    name: str
    components: object
    constraint_weights: dict = field(default_factory=lambda: {'constraint_component': 1.})

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('Give the cost function a nonempty name')
        if not callable(self.components):
            raise TypeError('components must be callable')
        weights = {name: float(value) for name, value in self.constraint_weights.items()}
        if any(not isinstance(name, str) or not name or not isfinite(value)
               for name, value in weights.items()):
            raise ValueError('Constraint weights must have names and finite numeric values')
        object.__setattr__(self, 'constraint_weights', MappingProxyType(weights))

    def __call__(self, state, metrics, cfg):
        metrics.update(call_cost(self.components, state, metrics, cfg))
        constraints = sum(weight * metrics[name] for name, weight in self.constraint_weights.items())
        return float(cfg.alpha * metrics['passenger_component']
                     + (1 - cfg.alpha) * metrics['operator_component'] + cfg.beta * constraints)


def load_cost_function(folder):
    path = Path(folder) / 'cost_function.pkl'
    return cloudpickle.loads(path.read_bytes()) if path.exists() else network_cost

def save_cost_function(folder, function):
    if function is not network_cost:
        (Path(folder) / 'cost_function.pkl').write_bytes(cloudpickle.dumps(function))

class NativeObjective(nn.Module):
    """Call the same per-city objective in construction and scalar search."""
    def __init__(self, base, cfg, function):
        super().__init__()
        self.base, self.cfg, self.function = base, cfg, function

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('base'), name)

    def forward(self, state, *args, **kwargs):
        result = self.base(state, *args, **kwargs)
        diameter = state.drive_times.flatten(1).max(1).values / 60
        total = result.total_demand.clamp_min(1e-12)
        served = (result.total_demand - result.unserved_demand).clamp_min(1e-12)
        att = result.total_demand_time / served / 60
        rtt = result.total_route_time / 60
        penalized = (result.total_demand_time / 60 + result.unserved_demand * 2 * diameter) / total
        unconnected = result.n_disconnected_demand_edges / state.n_demand_edges.clamp_min(1)
        routes = result.batch_routes
        valid = routes >= 0
        covered = (torch.nn.functional.one_hot(routes.clamp_min(0), state.max_n_nodes).bool()
                   & valid[..., None]).any(1).any(1).sum(1) / state.n_nodes
        metrics = dict(ATT=att, RTT=rtt, diameter=diameter, penalized_ATT=penalized,
                       passenger_component=penalized / diameter,
                       operator_component=rtt / (state.n_routes_to_plan * diameter),
                       unserved_fraction=result.unserved_demand / total,
                       unconnected_pair_fraction=unconnected,
                       constraint_component=unconnected + .1 * (unconnected > 0),
                       covered_stops_fraction=covered, route_count=valid.any(-1).sum(-1))
        for i, name in enumerate(('d0', 'd1', 'd2', 'un')):
            metrics[name] = 100 * result.trips_at_transfers[:, i] / total
        # Call the same Python function as the single-city evaluator. Scalar
        # values also allow ordinary Python conditionals in notebook objectives.
        columns = {key: value.detach().cpu().tolist() for key, value in metrics.items()}
        alphas = state.cost_weights['demand_time_weight'].expand(state.batch_size).detach().cpu().tolist()
        views = native_cost_states(state) if accepts_state(self.function) else [None] * state.batch_size
        values = [float(call_cost(self.function, view,
                                  {key: column[i] for key, column in columns.items()},
                                  replace(self.cfg, alpha=float(alphas[i]), n_nodes=int(state.n_nodes[i]),
                                          n_routes=int(state.n_routes_to_plan[i]))))
                  for i, view in enumerate(views)]
        cost = torch.as_tensor(values, dtype=result.cost.dtype, device=result.cost.device)
        if not torch.isfinite(cost).all():
            raise ValueError('Cost function must return a finite scalar for every city')
        return replace(result, _cost=cost)

def native_objective(base, cfg, function):
    return base if function is network_cost else NativeObjective(base, cfg, function)
