"""Разделение train/validation/test, сохранение и загрузка выборок."""
from torch.utils.data import Dataset
import numpy as np
import networkx as nx
from tndp.data.city import generate_city, make_city
import pandas as pd
from tqdm.auto import tqdm

class CityDataset(Dataset):
    """One item is a City: streets, coordinates, OD demand and shortest paths."""

    def __init__(self, cities):
        self.cities = list(cities)
        if not self.cities:
            raise ValueError('Dataset must contain at least one city')

    def __len__(self):
        return len(self.cities)

    def __getitem__(self, index):
        return self.cities[index]

def generate_dataset(size, n_nodes=15, seed=42, **graph_parameters):
    if size < 1:
        raise ValueError('Dataset size must be positive')
    return CityDataset(generate_city(n_nodes, seed + i, **graph_parameters)
                       for i in tqdm(range(size), desc='Генерация городов', unit='city', leave=False))

def create_datasets(cfg, train_size=64, val_size=8, test_size=4, **graph_parameters):
    """Independent, reproducible city sets for training, validation and test."""
    return {
        'train': generate_dataset(train_size, cfg.n_nodes, cfg.seed, **graph_parameters),
        'validation': generate_dataset(val_size, cfg.n_nodes, cfg.seed + train_size, **graph_parameters),
        'test': generate_dataset(test_size, cfg.n_nodes, cfg.seed + train_size + val_size, **graph_parameters),
    }

def load_datasets(path):
    """Restore the exact cities/splits saved by prepare_experiment()."""
    datasets = {}
    with np.load(path, allow_pickle=False) as saved:
        total = sum(len(saved[f'{split}_coords']) for split in ('train', 'validation', 'test'))
        with tqdm(total=total, desc='Загрузка городов', unit='city', leave=False) as progress:
            for split in ('train', 'validation', 'test'):
                cities = []
                for coords, streets, demand in zip(saved[f'{split}_coords'], saved[f'{split}_streets'],
                                                   saved[f'{split}_demand'], strict=True):
                    cities.append(make_city(coords, streets, demand))
                    progress.update()
                types_key = f'{split}_graph_types'
                if types_key in saved:
                    for city, kind in zip(cities, saved[types_key], strict=True):
                        city.graph.graph['graph_type'] = str(kind)
                datasets[split] = CityDataset(cities)
    return datasets

def save_datasets(datasets, path):
    """Save the exact cities/splits used by a training or evaluation run."""
    arrays = {}
    with tqdm(total=len(datasets) + 1, desc='Сохранение данных', unit='part', leave=False) as progress:
        for split, cities in datasets.items():
            arrays[f'{split}_graph_types'] = np.array([city.graph.graph.get('graph_type', 'external') for city in cities])
            arrays[f'{split}_coords'] = np.stack([city.coords for city in cities])
            arrays[f'{split}_streets'] = np.stack([nx.to_numpy_array(
                city.graph, nodelist=range(city.n_nodes), nonedge=np.inf) for city in cities])
            arrays[f'{split}_demand'] = np.stack([city.demand for city in cities])
            progress.update()
        np.savez_compressed(path, **arrays)
        progress.update()

def collect_data(config):
    data = config.data
    if data.source is None:
        datasets = create_datasets(config.problem, data.train_size, data.val_size, data.test_size,
                                   graph_type=data.graph_type, edge_keep_probability=data.edge_keep_probability,
                                   side_km=data.side_km, speed_kmh=data.speed_kmh)
    else:
        datasets = load_datasets(data.source)
    expected = dict(train=data.train_size, validation=data.val_size, test=data.test_size)
    if {name: len(cities) for name, cities in datasets.items()} != expected:
        raise ValueError('Dataset sizes differ from DatasetConfig')
    if any(city.n_nodes != config.problem.n_nodes for cities in datasets.values() for city in cities):
        raise ValueError('City sizes differ from ProblemConfig')
    return datasets

def dataset_table(datasets):
    return pd.DataFrame({'Города': {name: len(cities) for name, cities in datasets.items()}}).rename(
        index={'train': 'Обучение', 'validation': 'Валидация', 'test': 'Проверка'})
