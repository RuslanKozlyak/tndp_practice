"""Проверка готовой политики и методов поиска на Mumford 0."""
from dataclasses import replace
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from tndp.data.city import load_mumford0
from tndp.evaluation.tables import compact_table
from tndp.evaluation.methods import evaluate_rl, evaluate_lc, evaluate_evolution
from tndp.evaluation.pareto import evaluate_pareto

def _prepare_benchmark(config):
    benchmark = config.benchmark
    city = load_mumford0(benchmark.path)
    problem = replace(config.problem, n_nodes=benchmark.n_nodes, n_routes=benchmark.n_routes,
                      min_route_len=benchmark.min_route_len, max_route_len=benchmark.max_route_len)
    benchmark_config = replace(config, problem=problem,
                                output_dir=str(Path(config.output_dir) / 'mumford0'))
    return city, benchmark_config


def evaluate_benchmark_pareto(config, training, *, methods=('nea',)):
    """Sweep preferences on Mumford 0 with its own route-count and length limits."""
    city, benchmark_config = _prepare_benchmark(config)
    return evaluate_pareto([city], benchmark_config, training, methods=methods)


def evaluate_benchmark(config, training):
    city, benchmark_config = _prepare_benchmark(config)
    with tqdm(total=3, desc='Mumford 0', unit='stage') as progress:
        rl = evaluate_rl([city], benchmark_config, training)
        progress.update()
        lc = evaluate_lc([city], benchmark_config, training, rl)
        progress.update()
        evolution = evaluate_evolution([city], benchmark_config, training, lc)
        progress.update()
    result = dict(scores=pd.concat([rl['scores'], lc['scores'], evolution['scores']], ignore_index=True),
                  results={**rl['results'], **lc['results'], **evolution['results']}, city=city, metrics=config.metrics)
    compact_table(result).to_csv(Path(benchmark_config.output_dir) / 'summary.csv')
    return result
