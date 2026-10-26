"""Кривые ATT–RTT для RL, LC, EA и NEA при изменении веса стоимости."""
from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from tndp.search.inference import rollout, sample_from_policy, generate_routes

def nondominated(points):
    """Minimize both columns; equal points do not dominate each other."""
    values = np.asarray(points, dtype=float)
    if not len(values):
        return np.zeros(0, dtype=bool)
    return np.array([not np.any(np.all(values <= point, axis=1) & np.any(values < point, axis=1))
                     for point in values], dtype=bool)

def _mark_fronts(points):
    """Each city and method has its own front; methods never eliminate each other."""
    points['pareto'] = False
    for _, group in points.loc[points.feasible.astype(bool)].groupby(['city', 'model', 'method'], sort=False):
        points.loc[group.index, 'pareto'] = nondominated(group[['ATT', 'RTT']])
    return points


def pareto_summary(points):
    """Keep the single result per method, city and alpha without averaging."""
    keys = ['city', 'model', 'method', 'alpha']
    if points.duplicated(keys).any():
        raise ValueError('Expected one result per method, city and alpha')
    return _mark_fronts(points.drop(columns=['seconds'], errors='ignore').copy())


def _save_networks(folder, networks):
    """Publish a complete JSON file after every finished method, including interruptions."""
    temporary = folder / 'networks.json.tmp'
    temporary.write_text(json.dumps(networks, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(folder / 'networks.json')


def evaluate_pareto(cities, config, training, *, methods=('nea',)):
    """Sweep alpha for selected methods, using LC to initialize EA/NEA.

    Only requested methods are recorded. RL/LC preparation is shared when
    several methods are selected; selecting NEA never runs EA.
    """
    methods = (methods,) if isinstance(methods, str) else tuple(methods)
    methods = tuple(str(method).lower() for method in methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - {'rl', 'lc', 'ea', 'nea'}:
        raise ValueError('Choose distinct methods from rl, lc, ea, nea')
    rows, networks = [], []
    folder = Path(config.output_dir) / 'evaluations/pareto'
    folder.mkdir(parents=True, exist_ok=True)
    total = len(cities) * len(training['policies']) * len(config.pareto_weights) * len(methods)
    try:
        with tqdm(total=total, desc='Парето', unit='run') as progress:
            for index, city in enumerate(cities):
                for model, policy in training['policies'].items():
                    for alpha in config.pareto_weights:
                        problem = replace(config.problem, alpha=float(alpha))
                        seed = problem.seed + index
                        def record(method, result, started):
                            metrics = result['metrics']
                            rows.append(dict(city=index, model=model, method=method, alpha=float(alpha),
                                             seconds=perf_counter() - started,
                                             feasible=metrics['unconnected_pair_fraction'] <= 1e-12,
                                             **metrics))
                            networks.append(deepcopy(dict(
                                city=index, model=model, method=method, alpha=float(alpha),
                                routes=result['routes'], metrics=metrics)))
                            _save_networks(folder, networks)
                            progress.update()

                        greedy = None
                        if 'rl' in methods:
                            progress.set_postfix(city=index + 1, alpha=alpha, method='RL')
                            started = perf_counter()
                            greedy = rollout(policy, city, problem, seed=seed, deterministic=True,
                                             cost_function=training.get('cost_function'))
                            record('RL', greedy, started)

                        if set(methods) & {'lc', 'ea', 'nea'}:
                            stage = 'LC' if 'lc' in methods else 'LC (старт)'
                            progress.set_postfix(method=stage, city=index + 1, alpha=alpha)
                            started = perf_counter()
                            lc = sample_from_policy(policy, city, problem, n_samples=config.lc_samples,
                                                    seed=seed, greedy_result=greedy,
                                                    cost_function=training.get('cost_function'))
                            if 'lc' in methods:
                                record(f'LC{config.lc_samples}', lc, started)

                        for method, algorithm in (('EA', 'ga'), ('NEA', 'ga_rl')):
                            if method.lower() not in methods:
                                continue
                            progress.set_postfix(method=method, city=index + 1, alpha=alpha)
                            started = perf_counter()
                            result = generate_routes(city, problem, method=algorithm, policy=policy,
                                                     ga_cfg=config.ea, seed=seed, initial_routes=lc['routes'],
                                                     cost_function=training.get('cost_function'))
                            record(f'{method} (LC{config.lc_samples})', result, started)
    finally:
        # Preserve completed runs if a long sweep is interrupted.
        points = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['city', 'model', 'method', 'alpha', 'feasible', 'ATT', 'RTT', 'cost'])
        _mark_fronts(points).to_csv(folder / 'points.csv', index=False)
        pareto_summary(points).to_csv(folder / 'summary.csv', index=False)
        _save_networks(folder, networks)
    return points
