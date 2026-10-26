"""search / operators"""


MUTATIONS = ('replace', 'add', 'remove')

def valid_route(city, route, cfg):
    return (cfg.min_route_len <= len(route) <= cfg.max_route_len
            and len(set(route)) == len(route)
            and all(node in city.graph for node in route)
            and all(city.graph.has_edge(a, b) for a, b in zip(route, route[1:])))

def validate_network(city, routes, cfg, *, partial=False):
    if (len(routes) > cfg.n_routes if partial else len(routes) != cfg.n_routes):
        raise ValueError('Wrong number of routes')
    if any(not valid_route(city, route, cfg) for route in routes):
        raise ValueError('Routes must obey MIN/MAX, use street edges and contain no repeats')

def initial_paths(city, cfg):
    paths = [list(path) for path in city.paths.values() if valid_route(city, path, cfg)]
    if not paths:
        raise ValueError('No shortest path satisfies MIN/MAX. Reduce MIN or change the graph.')
    return paths

def random_network(city, cfg, rng):
    paths = initial_paths(city, cfg)
    return [paths[int(rng.integers(len(paths)))].copy() for _ in range(cfg.n_routes)]

def mutate_route(city, route, cfg, rng, kind):
    route = list(route)
    at_start = bool(rng.integers(2))
    if kind == 'add' and len(route) < cfg.max_route_len:
        end = route[0] if at_start else route[-1]
        choices = sorted(set(city.graph.neighbors(end)) - set(route))
        if choices:
            node = int(rng.choice(choices))
            return [node] + route if at_start else route + [node]
    elif kind == 'remove' and len(route) > cfg.min_route_len:
        return route[1:] if at_start else route[:-1]
    elif kind == 'replace':
        # Keep one old terminal. Choose a new terminal proportional to all
        # demand directly served along its shortest path, not just endpoint OD.
        end = route[0] if at_start else route[-1]
        paths = [path for (source, _), path in city.paths.items()
                 if source == end and valid_route(city, path, cfg)]
        if paths:
            weights = [sum(city.demand[a, b] for a in path for b in path) for path in paths]
            total = sum(weights)
            probabilities = [w / total for w in weights] if total else None
            return list(paths[int(rng.choice(len(paths), p=probabilities))])
    elif kind not in MUTATIONS:
        raise ValueError(f'Unknown mutation {kind!r}')
    return route

def mutate_network(city, routes, cfg, rng, kind, neural_builder=None):
    child = [list(route) for route in routes]
    index = int(rng.integers(len(child)))
    if kind == 'neural':
        if neural_builder is None:
            raise ValueError('Neural mutation requires a route builder')
        context = child[:index] + child[index + 1:]
        child[index] = list(neural_builder(context, int(rng.integers(2**31))))
    else:
        child[index] = mutate_route(city, child[index], cfg, rng, kind)
    validate_network(city, child, cfg)
    return child
