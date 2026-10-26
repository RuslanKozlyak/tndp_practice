"""Compare callable objectives on the same cities, policy weights and random seeds."""
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import json
import re

import pandas as pd

from tndp.costs.custom import save_cost_function
from tndp.costs.objective import get_cost_state
from tndp.costs.state import accepts_state, call_cost
from tndp.config import DEFAULT_METRICS
from tndp.data.city import City
from tndp.evaluation.methods import evaluate_rl, evaluate_lc, evaluate_evolution
from tndp.evaluation.tables import compact_table
from tndp.search.inference import save_result


def _named_functions(functions):
    if isinstance(functions, Mapping):
        pairs = list(functions.items())
    else:
        functions = [functions] if callable(functions) else list(functions)
        pairs = [(getattr(function, 'name', getattr(function, '__name__', type(function).__name__)), function)
                 for function in functions]
    if not pairs or any(not isinstance(name, str) or not name.strip() for name, _ in pairs):
        raise ValueError('Pass at least one named cost function')
    if len({name for name, _ in pairs}) != len(pairs):
        raise ValueError('Cost function names must be distinct; use a mapping for unnamed callables')
    if any(not callable(function) for _, function in pairs):
        raise TypeError('Each cost function must be callable')
    return pairs


def _report_metrics(stage, stage_name, cities, config, functions):
    """Measure every final network without changing its optimized cost."""
    needs_state = any(accepts_state(function) for _, function in functions)
    for (method, city), result in stage['results'].items():
        state = get_cost_state(cities[city], result['routes'], config.problem) if needs_state else None
        for name, function in functions:
            value = float(call_cost(function, state, dict(result['metrics']), config.problem))
            result['metrics'][name] = value
            rows = (stage['scores'].method == method) & (stage['scores'].city == city)
            stage['scores'].loc[rows, name] = value
        destination = Path(config.output_dir) / 'evaluations' / stage_name / method.replace(' ', '_') / f'city_{city}'
        save_result(result, destination)
    stage['metrics'] = tuple(dict.fromkeys((*config.metrics, *(name for name, _ in functions))))
    folder = Path(config.output_dir) / 'evaluations' / stage_name
    stage['scores'].to_csv(folder / 'metrics.csv', index=False)
    compact_table(stage).to_csv(folder / 'summary.csv')


def compare_cost_functions(cities, config, training, cost_functions, *,
                           methods=('rl', 'lc', 'ea', 'nea'), metrics=None, metric_functions=(), show=True):
    """Evaluate selected methods independently for each callable; return all networks.

    cost_functions accepts Python functions, callable objects or a name->callable mapping.
    No training occurs. LC and evolution reuse their own objective's RL/LC starts.
    metric_functions measures each final network, independently of its objective.
    """
    pairs = _named_functions(cost_functions)
    reporting = _named_functions(metric_functions) if metric_functions else []
    reserved = {'cost', 'C', 'method', 'model', 'city', 'alpha', 'seconds', 'function'}
    if any(name in reserved for name, _ in reporting):
        raise ValueError('Report metrics cannot replace cost or evaluation bookkeeping fields')
    methods = tuple(str(method).lower() for method in methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - {'rl', 'lc', 'ea', 'nea'}:
        raise ValueError('Choose distinct methods from rl, lc, ea, nea')
    cities = [cities] if isinstance(cities, City) else list(cities)
    if not cities or any(city.n_nodes != config.problem.n_nodes for city in cities):
        raise ValueError('Pass cities matching config.problem.n_nodes')
    requested = ((*config.metrics, *(name for name, _ in reporting)) if metrics is None else metrics)
    names = tuple('cost' if name == 'C' else name for name in requested)
    if metrics is None:
        names = tuple(dict.fromkeys(names))
    if not names or len(set(names)) != len(names):
        raise ValueError('Choose distinct metrics')
    folder = Path(config.output_dir) / 'evaluations/cost_functions'
    folder.mkdir(parents=True, exist_ok=True)
    evaluations, results, frames, networks, objective_folders = {}, {}, [], [], {}
    for index, (name, function) in enumerate(pairs):
        slug = re.sub(r'[^\w-]+', '_', name).strip('_') or 'cost'
        destination = folder / f'{index + 1:02d}_{slug}'
        destination.mkdir(parents=True, exist_ok=True)
        # A custom metric may belong to only one objective. Intermediate stages
        # summarize shared metrics; the combined table includes all requested ones.
        stage_metrics = tuple(metric for metric in names if metric in DEFAULT_METRICS) or ('cost',)
        current = replace(config, output_dir=str(destination), metrics=stage_metrics)
        local_training = {**training, 'cost_function': function}
        save_cost_function(destination, function)
        stages = []
        rl = None
        if 'rl' in methods:
            rl = evaluate_rl(cities, current, local_training)
            stages.append(('rl', rl))
        if set(methods) & {'lc', 'ea', 'nea'}:
            evolution_methods = tuple(method for method in methods if method in ('ea', 'nea'))
            description = (None if 'lc' in methods else
                           f'Инициализация {" / ".join(method.upper() for method in evolution_methods)}: LC{config.lc_samples}')
            lc = evaluate_lc(cities, current, local_training, rl, description=description)
            if 'lc' in methods:
                stages.append(('lc', lc))
            if evolution_methods:
                stages.append(('evolution', evaluate_evolution(cities, current, local_training, lc, methods=evolution_methods)))
        if reporting:
            for stage_name, stage in stages:
                _report_metrics(stage, stage_name, cities, current, reporting)
        scores = pd.concat([stage['scores'] for _, stage in stages], ignore_index=True)
        local_results = {key: value for _, stage in stages for key, value in stage['results'].items()}
        evaluations[name] = dict(scores=scores, results=local_results, metrics=names)
        objective_folders[name] = str(destination)
        frames.append(scores.assign(function=name))
        for (method, city), result in local_results.items():
            results[name, method, city] = result
            networks.append(dict(function=name, method=method, city=city,
                                 routes=result['routes'], metrics=result['metrics']))
    scores = pd.concat(frames, ignore_index=True)
    missing = set(names) - set(scores.select_dtypes(include='number').columns)
    if missing:
        raise ValueError(f'Unknown or nonnumeric metrics: {sorted(missing)}')
    for name, evaluation in evaluations.items():
        for metric in names:
            if metric not in evaluation['scores']:
                evaluation['scores'][metric] = float('nan')
        destination = Path(objective_folders[name])
        evaluation['scores'].to_csv(destination / 'metrics.csv', index=False)
        compact_table(evaluation, names).to_csv(destination / 'summary.csv')
    summary = pd.concat({name: compact_table(evaluation, names) for name, evaluation in evaluations.items()},
                        names=['Функция'])
    scores.to_csv(folder / 'metrics.csv', index=False)
    summary.to_csv(folder / 'summary.csv')
    (folder / 'networks.json').write_text(json.dumps(networks, ensure_ascii=False, indent=2), encoding='utf-8')
    comparison = dict(scores=scores, summary=summary, results=results, evaluations=evaluations,
                      cities=cities, metrics=names, output_dir=str(folder), objective_folders=objective_folders,
                      cost_functions=dict(pairs), figures=[])
    if show:
        from tndp.visualization import show_cost_comparison
        comparison['figures'] = show_cost_comparison(comparison)
    return comparison
