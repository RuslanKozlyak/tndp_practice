"""Город в минутах: уличная сеть, OD-спрос и генерация синтетических примеров."""
from dataclasses import dataclass
from pathlib import Path
import networkx as nx
import numpy as np
from scipy.spatial import Voronoi, distance_matrix

@dataclass
class City:
    coords: np.ndarray
    graph: nx.Graph
    demand: np.ndarray
    distances: np.ndarray
    paths: dict

    @property
    def n_nodes(self):
        return len(self.coords)

    @property
    def diameter(self):
        return max(float(self.distances.max()), 1e-6)

def make_city(coords, adjacency, demand):
    """Missing street edges are inf. Edit demand directly for classroom exercises."""
    coords = np.asarray(coords, dtype=float)
    adjacency = np.asarray(adjacency, dtype=float)
    demand = np.asarray(demand, dtype=float).copy()
    n = len(coords)
    if coords.shape != (n, 2) or adjacency.shape != (n, n) or demand.shape != (n, n):
        raise ValueError('Expected coords (N,2), adjacency (N,N), demand (N,N)')
    if not np.isfinite(coords).all() or not np.isfinite(demand).all() or np.any(demand < 0):
        raise ValueError('Coordinates and demand must be finite; demand must be nonnegative')
    if not np.allclose(adjacency, adjacency.T) or np.isnan(adjacency).any():
        raise ValueError('This lesson uses an undirected street graph')
    edges = [(i, j, float(adjacency[i, j])) for i in range(n) for j in range(i + 1, n)
             if np.isfinite(adjacency[i, j])]
    if any(w <= 0 for _, _, w in edges):
        raise ValueError('Street times must be positive')
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(edges)
    if not nx.is_connected(graph):
        raise ValueError('The street graph must be connected')
    np.fill_diagonal(demand, 0)
    distances = np.zeros((n, n))
    paths = {}
    for source, (lengths, source_paths) in nx.all_pairs_dijkstra(graph, weight='weight'):
        for target in range(n):
            distances[source, target] = lengths[target]
            if source != target:
                paths[source, target] = source_paths[target]
    return City(coords, graph, demand, distances, paths)

GRAPH_TYPES = ('knn', '4_grid', '8_grid', 'voronoi', 'mst')

def _grid(n, diagonal, rng):
    factors = [r for r in range(2, int(np.sqrt(n)) + 1) if n % r == 0]
    if not factors:
        raise ValueError('Grid graphs require a composite node count >= 4')
    rows = factors[-1]
    columns = n // rows
    extent = np.array([rows, columns]) / np.sqrt(n) + rng.uniform(-0.1, 0.1, 2)
    coords = np.array([(x, y) for x in np.linspace(-extent[0], extent[0], rows)
                       for y in np.linspace(-extent[1], extent[1], columns)])
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    offsets = [(1, 0), (0, 1)] + ([(1, 1), (1, -1)] if diagonal else [])
    for row in range(rows):
        for col in range(columns):
            for dr, dc in offsets:
                if row + dr < rows and 0 <= col + dc < columns:
                    graph.add_edge(row * columns + col, (row + dr) * columns + col + dc)
    return coords, graph

def _voronoi(n, rng):
    """Stops are cell vertices, NOT the input sites (which give a Delaunay graph)."""
    n_sites, step = max(4, n), max(n // 4, 1)
    for _ in range(4096):
        diagram = Voronoi(rng.uniform(-1, 1, (n_sites, 2)))
        inside = np.flatnonzero((np.abs(diagram.vertices) < 1).all(axis=1))
        if len(inside) != n:
            n_sites = max(4, n_sites + (step if len(inside) < n else -step))
            step = max(step // 2, 1)
            continue
        mapping = {old: new for new, old in enumerate(inside)}
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from((mapping[a], mapping[b]) for a, b in diagram.ridge_vertices
                             if a in mapping and b in mapping)
        if nx.is_connected(graph):
            return diagram.vertices[inside], graph
    raise RuntimeError('Could not sample a connected Voronoi graph of the requested size')

def _street_graph(n, kind, keep, rng):
    if kind == 'voronoi':
        return _voronoi(n, rng)  # Reference code does not thin Voronoi edges.
    grid = kind in ('4_grid', '8_grid')
    if grid:
        coords, base = _grid(n, kind == '8_grid', rng)
    for _ in range(4096):
        if not grid:
            coords = rng.uniform(-1, 1, (n, 2))
            distances = distance_matrix(coords, coords)
            if kind == 'mst':
                complete = nx.from_numpy_array(distances)
                graph = nx.minimum_spanning_tree(complete)
                # Density rule used in both reference repositories.
                target = min(n * (n - 1) // 2, max(n - 1, int(3.616 * n - 34.204)))
                for a, b in sorted(complete.edges, key=lambda edge: distances[edge]):
                    if graph.number_of_edges() >= target:
                        break
                    graph.add_edge(a, b)
                return coords, graph
            np.fill_diagonal(distances, np.inf)
            nearest = np.argsort(distances, axis=1)[:, :min(4, n - 1)]
            base = nx.Graph()
            base.add_nodes_from(range(n))
            base.add_edges_from((i, int(j)) for i, neighbors in enumerate(nearest) for j in neighbors)
        graph = base.copy()
        graph.remove_edges_from([edge for edge in base.edges if rng.random() > keep])
        if nx.is_connected(graph):
            return coords, graph
    raise RuntimeError('Could not sample connected streets; increase edge_keep_probability')

def generate_city(n_nodes=15, seed=42, *, graph_type='mixed',
                  edge_keep_probability=0.7, side_km=30.0, speed_kmh=54.0):
    """Holliday/ConnectPT graph families and symmetric uniform OD demand.

    mixed includes MST with extra short edges as in the two repositories;
    the paper itself lists only knn, the two grids and Voronoi. Disconnected
    samples are rejected, never repaired by silently inserting an MST.
    """
    if n_nodes < 2 or side_km <= 0 or speed_kmh <= 0 or not 0 < edge_keep_probability <= 1:
        raise ValueError('Require N >= 2, positive scale/speed and edge keep probability in (0,1]')
    if graph_type not in (*GRAPH_TYPES, 'mixed'):
        raise ValueError(f'graph_type must be mixed or one of {GRAPH_TYPES}')
    rng = np.random.default_rng(seed)
    if graph_type == 'mixed':
        choices = list(GRAPH_TYPES)
        if not any(n_nodes % r == 0 for r in range(2, int(np.sqrt(n_nodes)) + 1)):
            choices = [kind for kind in choices if 'grid' not in kind]
        if n_nodes < 4:
            choices.remove('voronoi')
        graph_type = str(rng.choice(choices))
    coords, graph = _street_graph(n_nodes, graph_type, edge_keep_probability, rng)
    coords = coords * side_km / 2
    distances = distance_matrix(coords, coords)
    adjacency = np.full((n_nodes, n_nodes), np.inf)
    np.fill_diagonal(adjacency, 0)
    for a, b in graph.edges:
        adjacency[a, b] = adjacency[b, a] = distances[a, b] * 60 / speed_kmh
    demand = np.triu(rng.uniform(60, 800, (n_nodes, n_nodes)), 1)
    city = make_city(coords, adjacency, demand + demand.T)
    city.graph.graph['graph_type'] = graph_type
    return city

def load_mandl(folder='benchmark/mandl'):
    """Read the three Mandl text files; their times are minutes."""
    folder = Path(folder)
    return make_city(np.loadtxt(folder / 'MandlCoords.txt', skiprows=1),
                     np.loadtxt(folder / 'MandlTravelTimes.txt'),
                     np.loadtxt(folder / 'MandlDemand.txt'))

def load_mumford0(folder='benchmark/mumford0'):
    """Load the original 30-stop instance; edge times are already minutes."""
    folder = Path(folder)
    city = make_city(np.loadtxt(folder / 'Coords.txt', skiprows=1),
                     np.loadtxt(folder / 'TravelTimes.txt'),
                     np.loadtxt(folder / 'Demand.txt'))
    if city.n_nodes != 30:
        raise ValueError('Mumford 0 must contain 30 stops')
    city.graph.graph['graph_type'] = 'mumford0'
    return city
