"""Построение маршрутной сети на одном городе готовой политикой."""
from dataclasses import replace
import pandas as pd
import torch
from tqdm.auto import tqdm
from tndp.costs.custom import native_objective
from tndp.costs.objective import evaluate_network, network_cost
from tndp.environment.construction import ConstructionEnv
from tndp.training.rollout import rollout
from tndp.training.settings import DEVICE
from tndp.data.graph import city_graph
from tndp.data.transforms import pipeline

class ConstructionPolicy:
    """Expose native route generation to the existing LC/NEA experiment interface."""
    def __init__(self, policy, cost, limits, native, rl, cost_function=network_cost):
        self.policy, self.cost, self.limits, self.native = policy.eval(), cost, limits, native
        self.cost_function = cost_function

    @torch.no_grad()
    def rollout(self, city, cfg, *, context=(), deterministic=True, seed=None,
                cost_function=None, progress=True):
        cost_function = self.cost_function if cost_function is None else cost_function
        graph = pipeline(self.native.dataset.evaluation_pipeline).graph(city_graph(city))
        # Graph policies transfer to other city/route sizes. Preferences must
        # enter the policy state as well as the final metric calculation.
        limits = replace(self.limits, n_routes=cfg.n_routes, min_route_len=cfg.min_route_len,
                         max_route_len=cfg.max_route_len)
        cost = self.cost
        if (cost.demand_time_weight != cfg.alpha or cost.route_time_weight != 1 - cfg.alpha
                or cost.constraint_violation_weight != cfg.beta
                or cost.avg_transfer_wait_time_s != cfg.transfer_penalty * 60):
            cost = cost.with_params(demand_time_weight=cfg.alpha, route_time_weight=1 - cfg.alpha,
                                    constraint_violation_weight=cfg.beta,
                                    avg_transfer_wait_time_s=cfg.transfer_penalty * 60).to(DEVICE)
        cost = native_objective(cost, cfg, cost_function)
        env = ConstructionEnv(cost, self.policy.feature_set, limits, self.policy.action_spec,
                               node_normalizer=self.policy.node_norm)
        initial = evaluate_network(city, context, cfg, cost_function)
        # The original environment can preload the other routes for an NEA mutation.
        env.reset([graph], cost.get_weights(), [list(context)] if context else None)
        rows, previous = [], float(env.result.cost[0])
        with torch.random.fork_rng(devices=[]), tqdm(
            total=cfg.n_routes, initial=int(env.state.n_finished_routes[0]),
            desc='Построение сети', unit='route', leave=False, disable=not progress
        ) as route_progress:
            torch.manual_seed(cfg.seed if seed is None else seed)
            while not env.is_done().all():
                trajectory = rollout(env, self.policy, greedy=deterministic, max_steps=1, record_states=False)
                current = float(env.result.cost[0])
                action = trajectory.outputs[0].action
                rows.append(dict(step=len(rows) + 1,
                                 action=f'{int(action.kind[0])}: {action.indices[0].tolist()}',
                                 cost=current, reward=previous - current))
                previous = current
                if progress:
                    route_progress.update(int(env.state.n_finished_routes[0]) - route_progress.n)
        routes = [[int(v) for v in route if v >= 0] for route in env.updated_network()[0].tolist()]
        routes = [route for route in routes if route]
        metrics = evaluate_network(city, routes, cfg, cost_function)
        return dict(routes=routes, metrics=metrics, history=pd.DataFrame(rows), initial_metrics=initial,
                    objective_evaluations=len(rows) + 3)
