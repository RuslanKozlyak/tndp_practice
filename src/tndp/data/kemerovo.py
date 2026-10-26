"""Kemerovo benchmark and the road geometry retained by graph simplification."""
from dataclasses import dataclass
from pathlib import Path
import json
import pickle

import numpy as np
from pyproj import Transformer

from tndp.data.city import City, make_city


@dataclass
class KemerovoData:
    city: City
    roads: object
    stops: np.ndarray
    crs: str


def load_kemerovo(folder='benchmark/kemerovo'):
    folder = Path(folder)
    coordinates = np.loadtxt(folder / 'KemCoords.txt', skiprows=1)
    streets = np.loadtxt(folder / 'KemTravelTimes.txt')
    demand = np.loadtxt(folder / 'KemDemand.txt')
    with (folder / 'kemerovo_simplified_graph.pkl').open('rb') as source:
        roads = pickle.load(source)
    n = len(coordinates)
    if set(roads) != set(range(n)):
        raise ValueError('Road node IDs must match the rows of the Kemerovo matrices')
    positions = np.array([[roads.nodes[i]['x'], roads.nodes[i]['y']] for i in range(n)])
    if not np.allclose(positions, coordinates, rtol=0, atol=.02):
        raise ValueError('Kemerovo coordinates and road node IDs disagree')
    matrix_edges = {tuple(edge) for edge in np.argwhere(np.triu(np.isfinite(streets), 1))}
    if matrix_edges != {tuple(sorted(edge)) for edge in roads.edges}:
        raise ValueError('Street matrix and saved road graph have different edges')
    # Input coordinates are UTM metres; the rest of the project uses kilometres.
    city = make_city(coordinates / 1000, streets, demand)
    city.graph.graph['graph_type'] = 'kemerovo'
    crs = str(roads.graph['crs'])
    data = json.loads((folder / 'kemerovo_stops.geojson').read_text(encoding='utf-8'))
    lonlat = np.array([feature['geometry']['coordinates'] for feature in data['features']
                      if feature['geometry']['type'] == 'Point'])
    transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    stops = np.column_stack(transformer.transform(lonlat[:, 0], lonlat[:, 1])) / 1000
    return KemerovoData(city, roads, stops, crs)


def road_edge_coordinates(data, source, target):
    """Return the saved road polyline, oriented source-to-target, in kilometres."""
    edge = data.roads[source][target]
    geometry = edge.get('geometry')
    if geometry is None or geometry.geom_type != 'LineString':
        raise ValueError(f'No road LineString for edge {source}–{target}')
    coordinates = np.asarray(geometry.coords, dtype=float) / 1000
    start = data.city.coords[source]
    if np.linalg.norm(coordinates[-1] - start) < np.linalg.norm(coordinates[0] - start):
        coordinates = coordinates[::-1]
    endpoints = data.city.coords[[source, target]]
    if np.max(np.linalg.norm(coordinates[[0, -1]] - endpoints, axis=1)) > .00002:
        raise ValueError(f'Road geometry does not meet the stops for edge {source}–{target}')
    return coordinates.copy()


def project_routes_to_roads(data, routes, output_dir):
    """Join stored edge geometries; keep segment boundaries and export WGS84 GeoJSON."""
    projected = []
    features = []
    transformer = Transformer.from_crs(data.crs, 'EPSG:4326', always_xy=True)
    for index, route in enumerate(routes):
        segments = [road_edge_coordinates(data, a, b) for a, b in zip(route, route[1:])]
        coordinates = np.concatenate([segment if i == 0 else segment[1:]
                                      for i, segment in enumerate(segments)]) if segments else data.city.coords[list(route)]
        projected.append(dict(route=index + 1, stops=list(route), segments=segments, coordinates=coordinates))
        lon, lat = transformer.transform(coordinates[:, 0] * 1000, coordinates[:, 1] * 1000)
        features.append(dict(type='Feature', properties=dict(route=index + 1, stops=list(route)),
                             geometry=dict(type='LineString', coordinates=np.column_stack((lon, lat)).tolist())))
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'routes_roads.geojson').write_text(
        json.dumps(dict(type='FeatureCollection', features=features), ensure_ascii=False), encoding='utf-8')
    return projected
