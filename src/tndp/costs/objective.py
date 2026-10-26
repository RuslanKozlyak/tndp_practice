"""Оценка одной маршрутной сети: стоимость и отчётные метрики."""
from itertools import combinations
import networkx as nx
import numpy as np
from tndp.search.operators import validate_network
from tndp.costs.state import make_cost_state, accepts_state, call_cost

def network_cost(metrics, cfg):
    return float(cfg.alpha * metrics['passenger_component']
                 + (1 - cfg.alpha) * metrics['operator_component']
                 + cfg.beta * metrics['constraint_component'])


def get_cost_metrics(city, routes, cfg):
    """Return the input dictionary available to a user-defined cost function."""
    metrics = evaluate_network(city, routes, cfg)
    metrics.pop('cost')
    return metrics


def get_cost_state(city, routes, cfg, *, transport=None, is_complete=None):
    """Inspect one city in minutes. A route list is complete when it has S routes.

    Pass is_complete=False when the last route is still being constructed.
    Unreachable passenger times are inf; their transfer count is zero.
    """
    transport = transport_state(city, routes, cfg) if transport is None else transport
    streets = nx.to_numpy_array(city.graph, nodelist=range(city.n_nodes), nonedge=np.inf)
    np.fill_diagonal(streets, 0)
    complete = len(routes) == cfg.n_routes if is_complete is None else is_complete
    return make_cost_state(routes, streets, city.demand, transport['times'],
                           transport['transfers'], complete)

def transport_state(city, routes, cfg):
    """Passenger paths from NetworkX, shared by the cost and observations.

    Nodes are (route, stop); changing routes costs transfer_penalty. Transfer
    counts describe the chosen fastest path, not a separate minimum-transfer
    optimization. Unreachable times are inf and transfer counts are zero.
    """
    graph = nx.Graph()
    by_stop = {}
    rtt = 0.0
    direct_times = np.full((city.n_nodes, city.n_nodes), np.inf)
    np.fill_diagonal(direct_times, 0)
    for route_id, route in enumerate(routes):
        for stop in route:
            state = (route_id, stop)
            graph.add_node(state)
            by_stop.setdefault(stop, []).append(state)
        for a, b in zip(route, route[1:]):
            time = city.graph[a][b]['weight']
            graph.add_edge((route_id, a), (route_id, b), weight=time)
            rtt += 2 * time
        cumulative = np.r_[0, np.cumsum([city.graph[a][b]['weight']
                                       for a, b in zip(route, route[1:])])]
        route_times = np.abs(cumulative[:, None] - cumulative[None, :])
        index = np.ix_(route, route)
        direct_times[index] = np.minimum(direct_times[index], route_times)
    for states in by_stop.values():
        graph.add_weighted_edges_from((a, b, cfg.transfer_penalty) for a, b in combinations(states, 2))
    times = np.full((city.n_nodes, city.n_nodes), np.inf)
    np.fill_diagonal(times, 0)
    transfers = np.zeros_like(times)
    for source, states in by_stop.items():
        distances, paths = nx.multi_source_dijkstra(graph, states, weight='weight')
        for state, time in distances.items():
            target = state[1]
            if time < times[source, target]:
                times[source, target] = time
                path = paths[state]
                transfers[source, target] = sum(a[0] != b[0] for a, b in zip(path, path[1:]))
    return dict(times=times, direct_times=direct_times, transfers=transfers,
                RTT=float(rtt), covered=list(by_stop))

def evaluate_network(city, routes, cfg, cost_function=network_cost, *, transport=None, is_complete=None):
    """ATT reports served demand; the objective imputes 2*diameter to the rest.

    As in ConnectPT, RTT counts both directions. Paper equations (5), (9)
    count one direction instead; this convention must be stated in comparisons.
    MIN/MAX and simple routes are enforced by the operators, so F_s is zero.
    """
    if city.n_nodes != cfg.n_nodes:
        raise ValueError('City size must equal cfg.n_nodes')
    validate_network(city, routes, cfg, partial=True)
    transport = transport_state(city, routes, cfg) if transport is None else transport
    times, rtt = transport['times'], transport['RTT']
    served = (city.demand > 0) & np.isfinite(times)
    total = float(city.demand.sum())
    served_demand = float(city.demand[served].sum())
    demand_time = float((city.demand[served] * times[served]).sum())
    att = demand_time / served_demand if served_demand else 0.0
    unserved = max(0.0, total - served_demand)
    penalized_att = (demand_time + unserved * 2 * city.diameter) / total if total else 0.0
    demand_pairs = int((city.demand > 0).sum())
    unconnected_pairs = 1 - float(served.sum()) / demand_pairs if demand_pairs else 0.0
    metrics = dict(ATT=att, RTT=float(rtt), diameter=float(city.diameter),
                   penalized_ATT=penalized_att,
                   unserved_fraction=unserved / total if total else 0.0,
                   unconnected_pair_fraction=unconnected_pairs,
                   constraint_component=unconnected_pairs + 0.1 * (unconnected_pairs > 0),
                   covered_stops_fraction=len(transport['covered']) / city.n_nodes,
                   route_count=len(routes),
                   passenger_component=penalized_att / city.diameter,
                   operator_component=rtt / (cfg.n_routes * city.diameter))
    # Demand shares by transfers along the fastest passenger path. The last
    # bucket includes unreachable trips and journeys requiring >2 transfers.
    for count in range(3):
        metrics[f'd{count}'] = 100 * float(city.demand[served & (transport['transfers'] == count)].sum()) / total if total else 0.
    metrics['un'] = 100 * float(city.demand[~served | (transport['transfers'] > 2)].sum()) / total if total else 0.
    # Notebook objectives may replace components or add named constraints.
    # Keep these fields in the result so they can also be inspected in tables.
    state = (get_cost_state(city, routes, cfg, transport=transport, is_complete=is_complete)
             if accepts_state(cost_function) else None)
    metrics['cost'] = float(call_cost(cost_function, state, metrics, cfg))
    if not np.isfinite(metrics['cost']):
        raise ValueError('network_cost must return a finite number')
    return metrics
