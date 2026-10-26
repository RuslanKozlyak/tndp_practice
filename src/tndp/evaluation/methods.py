"""Запуск RL, LC, EA и NEA на одинаковых тестовых городах."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import pandas as pd
from tqdm.auto import tqdm
from tndp.search.inference import rollout, sample_from_policy, generate_routes, save_result
from tndp.evaluation.tables import compact_table

def _record(results, rows, output, variant, model, index, result, seconds, cfg):
    results[variant, index] = result
    rows.append(dict(method=variant, model=model, city=index, alpha=cfg.alpha,
                     seconds=seconds, **result['metrics']))
    save_result(result, Path(output) / variant.replace(' ', '_') / f'city_{index}',
                dict(problem=asdict(cfg), method=variant, model=model))

def _finish(config, stage, results, rows):
    scores = pd.DataFrame(rows)
    folder = Path(config.output_dir) / 'evaluations' / stage
    folder.mkdir(parents=True, exist_ok=True)
    scores.to_csv(folder / 'metrics.csv', index=False)
    result = dict(results=results, scores=scores, metrics=config.metrics)
    compact_table(result).to_csv(folder / 'summary.csv')
    return result

def evaluate_rl(cities, config, training):
    results, rows = {}, []
    output = Path(config.output_dir) / 'evaluations/rl'
    for index, city in tqdm(enumerate(cities), total=len(cities), desc='RL', unit='city'):
        for model, policy in training['policies'].items():
            started = perf_counter()
            result = rollout(policy, city, config.problem, seed=config.problem.seed + index, deterministic=True,
                             cost_function=training.get('cost_function'))
            _record(results, rows, output, 'RL', model, index,
                    result, perf_counter() - started, config.problem)
    return _finish(config, 'rl', results, rows)

def evaluate_lc(cities, config, training, rl_results=None, *, description=None):
    results, rows = {}, []
    output = Path(config.output_dir) / 'evaluations/lc'
    for index, city in tqdm(enumerate(cities), total=len(cities),
                            desc=description or f'LC{config.lc_samples}', unit='city'):
        for model, policy in training['policies'].items():
            greedy = None if rl_results is None else rl_results['results']['RL', index]
            started = perf_counter()
            result = sample_from_policy(policy, city, config.problem, n_samples=config.lc_samples,
                                        seed=config.problem.seed + index, greedy_result=greedy,
                                        cost_function=training.get('cost_function'))
            _record(results, rows, output, f'LC{config.lc_samples}', model, index,
                    result, perf_counter() - started, config.problem)
    return _finish(config, 'lc', results, rows)

def evaluate_evolution(cities, config, training, lc_results, *, methods=('ea', 'nea')):
    """EA and NEA begin with clones of the same previously computed LC network."""
    algorithms = {'ea': ('EA', 'ga'), 'nea': ('NEA', 'ga_rl')}
    if not methods or len(set(methods)) != len(methods) or any(name not in algorithms for name in methods):
        raise ValueError('Evolution methods must be a nonempty selection of ea and nea')
    results, rows = {}, []
    output = Path(config.output_dir) / 'evaluations/evolution'
    description = ' / '.join(algorithms[name][0] for name in methods)
    for index, city in tqdm(enumerate(cities), total=len(cities), desc=description, unit='city'):
        for model, policy in training['policies'].items():
            initial = lc_results['results'][f'LC{config.lc_samples}', index]['routes']
            for method, algorithm in (algorithms[name] for name in methods):
                started = perf_counter()
                result = generate_routes(city, config.problem, method=algorithm, policy=policy,
                                         ga_cfg=config.ea, seed=config.problem.seed + index, initial_routes=initial,
                                         cost_function=training.get('cost_function'))
                _record(results, rows, output, f'{method} (LC{config.lc_samples})', model, index,
                        result, perf_counter() - started, config.problem)
    return _finish(config, 'evolution', results, rows)
