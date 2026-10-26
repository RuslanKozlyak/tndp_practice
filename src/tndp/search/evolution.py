"""EA и NEA: популяция маршрутных сетей, мутации и отбор."""
from time import perf_counter
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tndp.costs.objective import evaluate_network, network_cost
from tndp.search.operators import mutate_network, random_network, validate_network

def run_ga(city, cfg, ga_cfg, *, cost_function=network_cost,
           neural_builder=None, seed=None, initial_routes=None, initial_population=None, progress=True):
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    population = initialize_population(city, cfg, ga_cfg.population_size, rng,
                                       initial_routes, initial_population)
    starting_population = [[r.copy() for r in routes] for routes in population]

    evaluations = 0

    def cost(routes):
        nonlocal evaluations
        evaluations += 1
        return evaluate_network(city, routes, cfg, cost_function)['cost']

    costs = np.array([cost(routes) for routes in population])
    best_index = int(costs.argmin())
    best_cost, best_routes = float(costs[best_index]), population[best_index]
    starting_routes = [r.copy() for r in best_routes]

    history, mutations = [], []
    stats, recruiters, followers = [], 0, 0
    nevals = evaluations
    with tqdm(total=ga_cfg.generations, desc="NEA" if neural_builder else "EA",
              unit="gen", leave=False, disable=not progress) as generations:
        for generation in range(ga_cfg.generations + 1):
            history.append(generation_metrics(generation, population, costs, best_cost,
                                               evaluations, nevals, stats, recruiters, followers))
            if generation == ga_cfg.generations:
                break

            previous_evaluations = evaluations
            stats = improve_population(city, cfg, ga_cfg, population, costs, cost,
                                       rng, neural_builder, generation + 1)
            mutations.extend(stats)
            nevals = evaluations - previous_evaluations

            index = int(costs.argmin())
            if costs[index] < best_cost:
                best_cost, best_routes = float(costs[index]), population[index]
            # Stochastic recruitment can discard even the best bee: archive it first.
            sources, mask = recruit(costs, rng)
            recruiters = int(mask.sum())
            followers = int((sources != np.arange(len(costs))).sum())
            population = [[r.copy() for r in population[i]] for i in sources]
            costs = costs[sources].copy()
            generations.update()
    routes = [r.copy() for r in best_routes]
    result = dict(routes=routes, metrics=evaluate_network(city, routes, cfg, cost_function),
                  history=pd.DataFrame(history), mutations=pd.DataFrame(mutations),
                  initial_population=starting_population,
                  initial_metrics=evaluate_network(city, starting_routes, cfg, cost_function),
                  objective_evaluations=evaluations + 2)
    return result

def network_key(routes):
    """Line order and direction do not matter in this undirected task."""
    return tuple(sorted(min(tuple(r), tuple(reversed(r))) for r in routes))

def recruit(costs, rng):
    """Return the network each bee copies, and the recruiter mask.

    Quality q=(worst-cost)/(worst-best). Follow probability is exp(-q).
    Recruiters keep their networks; followers choose proportional to quality.
    Equal costs or no recruiters: everyone keeps their own network.
    """
    costs = np.asarray(costs, dtype=float)
    sources = np.arange(len(costs))
    spread = np.ptp(costs)
    if spread == 0:
        return sources, np.zeros(len(costs), dtype=bool)
    quality = (costs.max() - costs) / spread
    recruiters = rng.random(len(costs)) > np.exp(-quality)
    if recruiters.any():
        weights = quality * recruiters
        sources[~recruiters] = rng.choice(len(costs), size=(~recruiters).sum(),
                                         p=weights / weights.sum())
    return sources, recruiters

def initialize_population(city, cfg, size, rng, initial_routes=None, initial_population=None):
    """Use independent heuristic networks, clones of one network, or an exact population."""
    if initial_routes is not None and initial_population is not None:
        raise ValueError('Provide initial_routes or initial_population, not both')
    if initial_population is not None:
        if len(initial_population) != size:
            raise ValueError('initial_population must match population_size')
        for routes in initial_population:
            validate_network(city, routes, cfg)
        return [[list(route) for route in routes] for routes in initial_population]
    if initial_routes is not None:
        validate_network(city, initial_routes, cfg)
        return [[list(route) for route in initial_routes] for _ in range(size)]
    return [random_network(city, cfg, rng) for _ in range(size)]

def improve_population(city, cfg, ga_cfg, population, costs, cost,
                       rng, neural_builder, generation):
    """E rounds of mutation; each child competes only with its own parent."""
    rebuild = 'neural' if neural_builder is not None else 'replace'
    kinds = (rebuild, 'add', 'remove')
    stats = {kind: dict(generation=generation, kind=kind, attempts=0,
                       changed=0, improved=0, seconds=0.0) for kind in kinds}
    for _ in range(ga_cfg.mutation_steps):
        # Equal type-1/type-2 groups, randomly reassigned every round.
        for position, index in enumerate(rng.permutation(len(population))):
            if rng.random() >= ga_cfg.mutation_probability:
                continue
            kind = (rebuild if position < len(population) // 2 else
                    'remove' if rng.random() < ga_cfg.shorten_probability else 'add')
            parent = population[index]
            record = stats[kind]
            started = perf_counter()
            child = mutate_network(city, parent, cfg, rng, kind, neural_builder)
            record['attempts'] += 1
            if network_key(child) != network_key(parent):
                child_cost = cost(child)
                record['changed'] += 1
                # Each mutant competes with its own parent.
                if child_cost < costs[index]:
                    population[index], costs[index] = child, child_cost
                    record['improved'] += 1
            record['seconds'] += perf_counter() - started
    return list(stats.values())

def generation_metrics(generation, population, costs, best_cost, evaluations, nevals,
                       stats, recruiters, followers):
    return dict(
        generation=generation, cost=float(costs.min()), best_cost=best_cost,
        mean_cost=float(costs.mean()), nevals=nevals, cumulative_evaluations=evaluations,
        unique_networks=len({network_key(routes) for routes in population}),
        changed_offspring=sum(row['changed'] for row in stats),
        improved_offspring=sum(row['improved'] for row in stats),
        neural_attempts=sum(row['attempts'] for row in stats if row['kind'] == 'neural'),
        neural_improved=sum(row['improved'] for row in stats if row['kind'] == 'neural'),
        recruiters=recruiters, followers=followers)
