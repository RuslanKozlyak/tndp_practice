"""RL и LC: построение сети и выбор лучшей из нескольких генераций."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tndp.search.evolution import run_ga
from tndp.costs.objective import network_cost

def rollout(policy, city, cfg, *, context=(), deterministic=True, seed=None,
            cost_function=None, progress=True):
    """Construct one network with a trained policy; do not update its weights."""
    cost_function = getattr(policy, 'cost_function', network_cost) if cost_function is None else cost_function
    return policy.rollout(city, cfg, context=context, deterministic=deterministic,
                           seed=seed, cost_function=cost_function, progress=progress)

def sample_from_policy(policy, city, cfg, *, n_samples=100, seed=None,
                       cost_function=None, greedy_result=None, progress=True):
    """LC-k: select by cost from one greedy and k-1 stochastic networks.

    Sequential rollouts keep the lesson simple. No training occurs here. All
    sampled routes, costs and seeds are returned for saving and inspection.
    """
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 1:
        raise ValueError('n_samples must be a positive integer')
    seed = cfg.seed if seed is None else seed
    rng = np.random.default_rng(seed)
    samples, rows, best = [], [], None
    evaluations = 0
    for index in tqdm(range(n_samples), desc=f'LC{n_samples}', unit='sample', leave=False, disable=not progress):
        deterministic = index == 0
        sample_seed = seed if deterministic else int(rng.integers(2**31))
        if deterministic and greedy_result is not None:
            candidate = deepcopy(greedy_result)
        else:
            candidate = rollout(policy, city, cfg, deterministic=deterministic, seed=sample_seed,
                                cost_function=cost_function, progress=False)
            evaluations += candidate.get('objective_evaluations', 0)
        samples.append(dict(seed=sample_seed, deterministic=deterministic,
                            routes=candidate['routes'], metrics=candidate['metrics']))
        if best is None or candidate['metrics']['cost'] < best['metrics']['cost']:
            best = candidate
        rows.append(dict(sample=index + 1, seed=sample_seed, deterministic=deterministic,
                         cost=candidate['metrics']['cost'],
                         best_cost=best['metrics']['cost']))
    return dict(routes=best['routes'], metrics=best['metrics'], history=pd.DataFrame(rows),
                samples=samples, objective_evaluations=evaluations)

def generate_routes(city, cfg, *, method='ga', ga_cfg=None,
                    policy=None, deterministic=True, seed=None,
                    cost_function=None,
                    n_samples=1, initial_routes=None, initial_population=None, progress=True):
    cost_function = getattr(policy, 'cost_function', network_cost) if cost_function is None else cost_function
    if method not in ('ga', 'rl', 'ga_rl'):
        raise ValueError('Method must be ga, rl or ga_rl')
    if method in ('rl', 'ga_rl') and policy is None:
        raise ValueError('Train or load a policy first for rl / ga_rl')
    if method == 'rl':
        if initial_routes is not None or initial_population is not None:
            raise ValueError('Pure RL constructs from empty; initialization is only for ga / ga_rl')
        if n_samples != 1:
            return sample_from_policy(policy, city, cfg, n_samples=n_samples, seed=seed,
                                       cost_function=cost_function, progress=progress)
        return rollout(policy, city, cfg, deterministic=deterministic, seed=seed,
                       cost_function=cost_function, progress=progress)
    if ga_cfg is None:
        raise ValueError('Provide GAConfig for ga / ga_rl')
    neural_builder = None
    mutation_evaluations = 0
    if method == 'ga_rl':
        def neural_builder(context, mutation_seed):
            nonlocal mutation_evaluations
            result = rollout(policy, city, cfg, context=context, deterministic=False,
                             seed=mutation_seed, cost_function=cost_function, progress=False)
            mutation_evaluations += result['objective_evaluations']
            return result['routes'][-1]
    result = run_ga(city, cfg, ga_cfg, cost_function=cost_function,
                    neural_builder=neural_builder, seed=seed, initial_routes=initial_routes,
                    initial_population=initial_population, progress=progress)
    result['objective_evaluations'] += mutation_evaluations
    return result

def save_result(result, output_dir, settings=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {'routes': result['routes'], 'metrics': result['metrics']}
    payload['objective_evaluations'] = result.get('objective_evaluations', 0)
    if 'initial_metrics' in result:
        payload['initial_metrics'] = result['initial_metrics']
    if 'initial_population' in result:
        (output_dir / 'initial_population.json').write_text(
            json.dumps(result['initial_population'], indent=2), encoding='utf-8')
    (output_dir / 'routes.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    result['history'].to_csv(output_dir / 'search.csv', index=False)
    if 'mutations' in result:
        result['mutations'].to_csv(output_dir / 'mutations.csv', index=False)
    if 'samples' in result:
        (output_dir / 'samples.json').write_text(json.dumps(result['samples'], indent=2), encoding='utf-8')
    if 'sampling_history' in result:
        result['sampling_history'].to_csv(output_dir / 'sampling.csv', index=False)
    if settings is not None:
        (output_dir / 'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
