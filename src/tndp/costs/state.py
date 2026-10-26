"""Read-only, single-city inputs for user-defined objectives; times are minutes."""
from dataclasses import dataclass
from functools import lru_cache
from inspect import signature

import numpy as np


def readonly(values, dtype=None):
    """Own immutable bytes: neither assignment nor setflags can alter the source."""
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class CostState:
    routes: tuple[tuple[int, ...], ...]
    street_times: np.ndarray
    demand: np.ndarray
    travel_times: np.ndarray
    transfers: np.ndarray
    route_times: np.ndarray
    is_complete: bool


def make_cost_state(routes, street_times, demand, travel_times, transfers, is_complete):
    routes = tuple(tuple(int(stop) for stop in route) for route in routes)
    streets = readonly(street_times, float)
    route_times = [sum(streets[a, b] for a, b in zip(route, route[1:])) for route in routes]
    return CostState(routes, streets, readonly(demand, float), readonly(travel_times, float),
                     readonly(transfers, int), readonly(route_times, float), bool(is_complete))


def _accepts_state(function):
    """Keep existing two-argument callbacks usable without swallowing user errors."""
    parameters = signature(function)
    try:
        parameters.bind(None, None, None)
    except TypeError:
        parameters.bind(None, None)
        return False
    return True


_cached_accepts_state = lru_cache(maxsize=128)(_accepts_state)


def accepts_state(function):
    # Callable dataclass instances may deliberately be unhashable.
    try:
        hash(function)
    except TypeError:
        return _accepts_state(function)
    return _cached_accepts_state(function)


def call_cost(function, state, metrics, cfg):
    return function(state, metrics, cfg) if accepts_state(function) else function(metrics, cfg)


def native_cost_states(state):
    """Reuse prepared batch matrices; do not compute passenger paths again."""
    streets = state.street_adj.detach().cpu().numpy() / 60
    demand = state.demand.detach().cpu().numpy()
    travel = state.transit_times.detach().cpu().numpy() / 60
    transfers = state.n_transfers.detach().cpu().numpy()
    sizes = state.n_nodes.detach().cpu().tolist()
    done = state.is_done().detach().cpu().tolist()
    routes = [[route.detach().cpu().tolist() for route in network] for network in state.routes]
    current = state.current_routes.detach().cpu().tolist()
    for i, n in enumerate(sizes):
        # The internal passenger evaluator ignores a one-stop current route;
        # the public view still exposes it as unfinished work with zero time.
        partial = [stop for stop in current[i] if stop >= 0]
        if len(partial) == 1:
            routes[i].append(partial)
        travel_i = travel[i, :n, :n].copy()
        travel_i[~state.has_path[i, :n, :n].detach().cpu().numpy()] = np.inf
        np.fill_diagonal(travel_i, 0)
        yield make_cost_state(routes[i], streets[i, :n, :n], demand[i, :n, :n], travel_i,
                              transfers[i, :n, :n], done[i])
