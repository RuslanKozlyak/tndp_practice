"""Графики стоимости, Парето-фронты и карты маршрутов."""
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import json
from math import isclose

METHOD_STYLES = {
    'RL': ('#777777', 'v', ':'),
    'LC': ('#2ca02c', 's', '--'),
    'EA': ('#ff7f0e', 'x', '--'),
    'NEA': ('#1f77b4', 'o', '-'),
}


def _method_style(method):
    return METHOD_STYLES['LC' if method.startswith('LC') else method.split()[0]]


def plot_routes(ax, city, routes=(), title='Маршруты'):
    nx.draw_networkx(city.graph, pos=dict(enumerate(city.coords)), ax=ax,
                     node_size=100, font_size=8, node_color='lightgray', edge_color='0.8')
    for index, route in enumerate(routes):
        coords = city.coords[route]
        ax.plot(coords[:, 0], coords[:, 1], '-o', linewidth=2.5, markersize=4,
                label=f'Маршрут {index + 1}')
    ax.set_title(title)
    ax.set_aspect('equal')
    if routes:
        ax.legend(fontsize=8)

def show_cost_comparison(comparison, *, cities=None):
    """Display the per-objective table and separate route grids for the chosen cities."""
    from IPython.display import display
    from textwrap import fill
    display(comparison['summary'])
    indices = range(len(comparison['cities'])) if cities is None else cities
    functions = list(comparison['evaluations'])
    methods = list(dict.fromkeys(comparison['scores'].method))
    figures = []
    for index in indices:
        city = comparison['cities'][index]
        columns = min(2, len(functions))
        rows_per_method = (len(functions) + columns - 1) // columns
        rows = len(methods) * rows_per_method
        fig, axes = plt.subplots(rows, columns, squeeze=False,
                                 figsize=(5.5 * columns, 4.5 * rows), constrained_layout=True)
        for method_index, method in enumerate(methods):
            for function_index, function in enumerate(functions):
                row = method_index * rows_per_method + function_index // columns
                column = function_index % columns
                result = comparison['results'][function, method, index]
                metrics = result['metrics']
                title = (f'{fill(function, width=32)}\n{method}\n'
                         f'ATT = {metrics["ATT"]:.1f} мин · RTT = {metrics["RTT"]:.1f} мин')
                ax = axes[row, column]
                plot_routes(ax, city, result['routes'], title)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
                ax.set_axis_off()
            for unused in range(len(functions), rows_per_method * columns):
                axes[method_index * rows_per_method + unused // columns, unused % columns].remove()
        fig.suptitle(f'Город {index + 1} · сравнение функций стоимости')
        fig.savefig(Path(comparison['output_dir']) / f'routes_city_{index}.png', dpi=150)
        plt.show()
        figures.append(fig)
    return figures


def show_city(city):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    plot_routes(axes[0], city, title='Уличный граф')
    image = axes[1].imshow(city.demand, cmap='Blues')
    axes[1].set(xlabel='Куда', ylabel='Откуда', title='OD-спрос')
    fig.colorbar(image, ax=axes[1], label='Поездки')
    plt.show()
    return fig

def show_dataset_examples(cities, output_dir, n_examples=4):
    """Show training cities, preferring examples from distinct graph families."""
    if n_examples < 1 or not len(cities):
        raise ValueError('Choose at least one city and a positive n_examples')
    selected, families = [], set()
    for index, city in enumerate(cities):
        family = city.graph.graph.get('graph_type', 'external')
        if family not in families:
            selected.append(index)
            families.add(family)
        if len(selected) == n_examples:
            break
    selected += [index for index in range(len(cities)) if index not in selected][
        :max(0, n_examples - len(selected))]
    columns = min(2, len(selected))
    rows = (len(selected) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows),
                             squeeze=False, constrained_layout=True)
    for ax, index in zip(axes.flat, selected):
        city = cities[index]
        family = city.graph.graph.get('graph_type', 'external')
        title = (f'train[{index}] · {family}\n'
                 f'{city.n_nodes} остановок · {city.graph.number_of_edges()} связей')
        plot_routes(ax, city, title=title)
        ax.set(xlabel='x, км', ylabel='y, км')
    for ax in list(axes.flat)[len(selected):]:
        ax.remove()
    folder = Path(output_dir) / 'data'
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / 'training_graphs.png', dpi=150)
    plt.show()
    return fig


def show_training(training, output_dir):
    from pathlib import Path
    from matplotlib.ticker import MaxNLocator
    if not training['histories']:
        return None
    history, validation = training['histories']['gat'], training['validations']['gat']
    curve = history.groupby('ppo_updates').cost.mean().dropna().rolling(20, min_periods=1).mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(curve.index, curve, color='#54a24b', linewidth=2)
    axes[1].plot(validation.ppo_updates, validation.cost, 'o-', color='#54a24b', linewidth=2, markersize=4)
    for ax, title in zip(axes, ('Обучение', 'Валидация')):
        ax.set(title=title, xlabel='Обновления PPO', ylabel='Стоимость C ↓')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
    fig.savefig(Path(output_dir) / 'training.png', dpi=150)
    plt.show()
    return fig

def show_method_routes(evaluation, city, output_dir, *, city_index=0):
    """Compare distinct networks on one city, grouping identical method results."""
    networks = {}
    for (method, index), result in evaluation['results'].items():
        if index != city_index:
            continue
        routes = tuple(sorted(min(tuple(route), tuple(reversed(route)))
                              for route in result['routes']))
        if routes not in networks:
            networks[routes] = ([], result)
        networks[routes][0].append(method)
    results = list(networks.values())
    if not results:
        raise ValueError(f'No results for city {city_index}')
    columns = min(2, len(results))
    rows = (len(results) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 4.5 * rows),
                             squeeze=False, constrained_layout=True, sharex=True, sharey=True)
    for ax, (methods, result) in zip(axes.flat, results):
        metrics = result['metrics']
        plot_routes(ax, city, result['routes'],
                    f'{" = ".join(methods)}\nC = {metrics["cost"]:.3f} · ATT = {metrics["ATT"]:.1f} мин · RTT = {metrics["RTT"]:.1f} мин')
        ax.title.set_color(_method_style(methods[0])[0] if len(methods) == 1 else '0.25')
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_axis_off()
    for ax in list(axes.flat)[len(results):]:
        ax.remove()
    alpha = evaluation['scores'].iloc[0].alpha
    fig.suptitle(f'Тестовый город {city_index + 1} · одинаковый α = {alpha:g}')
    folder = Path(output_dir) / 'evaluations/comparison'
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f'routes_city_{city_index}.png', dpi=150)
    plt.show()
    return fig


def show_evolution(evaluation, output_dir, *, initial=None):
    """One cost curve per EA/NEA method, averaged over the same cities."""
    from pathlib import Path
    import pandas as pd
    frames = [result['history'][['generation', 'best_cost']].assign(method=method)
              for (method, _), result in evaluation['results'].items()]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    if frames:
        means = pd.concat(frames).groupby(['method', 'generation'], sort=False).best_cost.mean()
        for method in means.index.get_level_values(0).unique():
            curve = means.loc[method]
            color, marker, line = _method_style(method)
            ax.plot(curve.index, curve, label=method, color=color, marker=marker,
                    linestyle=line, linewidth=2, markersize=4)
        if initial is not None:
            ax.axhline(initial['scores'].cost.mean(), color=METHOD_STYLES['LC'][0],
                       linestyle=':', label='Начальная сеть LC')
        ax.legend()
    ax.set(xlabel='Поколение', ylabel='Стоимость C ↓',
           title=f'Среднее по тестовым городам: {evaluation["scores"].city.nunique()}')
    ax.grid(alpha=.2)
    fig.savefig(Path(output_dir) / 'evaluations/evolution/curves.png', dpi=150)
    plt.show()
    return fig

def show_pareto(points, output_dir, title=None):
    """Show each method's nondominated front and its other feasible solutions."""
    from pathlib import Path
    from tndp.evaluation.pareto import pareto_summary
    summary = pareto_summary(points)
    cities = sorted(points.city.unique())
    if not cities:
        return None
    columns = min(2, len(cities))
    fig, axes = plt.subplots((len(cities) + columns - 1) // columns, columns,
                             figsize=(6 * columns, 4.5 * ((len(cities) + columns - 1) // columns)),
                             squeeze=False, constrained_layout=True)
    for ax, city in zip(axes.flat, cities):
        group = summary[summary.city == city]
        single_method = group.method.nunique() == 1
        has_dominated = False
        for method, curve in group.groupby('method', sort=False):
            curve = curve.loc[curve.feasible]
            front = curve.loc[curve.pareto].sort_values('ATT')
            other = curve.loc[~curve.pareto]
            color, marker, line = _method_style(method)
            ax.plot(front.ATT, front.RTT, color=color, marker=marker, linestyle=line,
                    linewidth=1.6, markersize=5, label=method)
            if not other.empty:
                ax.scatter(other.ATT, other.RTT, facecolors='none', edgecolors=color,
                           s=45, alpha=.45)
                has_dominated = True
            labelled = curve if single_method else curve.loc[curve.alpha.isin([0., 1.])]
            for (att, rtt), same_point in labelled.groupby(['ATT', 'RTT']):
                label = 'α=' + ', '.join(f'{alpha:g}' for alpha in sorted(same_point.alpha))
                ax.annotate(label, (att, rtt), xytext=(6, 6), textcoords='offset points',
                            fontsize=8, color=color)
        if has_dominated:
            ax.scatter([], [], facecolors='none', edgecolors='0.5', s=45,
                       label='Доминируемое решение')
        if group.feasible.any():
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, 'Нет полностью связных сетей', ha='center', transform=ax.transAxes)
        ax.set(xlabel='ATT, мин ↓', ylabel='RTT, мин ↓',
               title=title if title and len(cities) == 1 else f'Город {city + 1}')
        excluded = int((~group.feasible).sum())
        if excluded:
            ax.text(.98, .02, f'Несвязных решений вне графика: {excluded}',
                    ha='right', transform=ax.transAxes, fontsize=8, color='0.4')
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(cities):]:
        ax.remove()
    folder = Path(output_dir) / 'evaluations/pareto'
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / 'fronts.png', dpi=150)
    plt.show()
    return fig

def _saved_benchmark_pareto(config):
    folder = Path(config.output_dir) / 'mumford0/evaluations/pareto'
    path = folder / 'networks.json'
    if not path.is_file():
        raise FileNotFoundError('Сначала выполните ячейку расчёта Парето на Mumford 0.')
    networks = json.loads(path.read_text(encoding='utf-8'))
    if not networks:
        raise ValueError('Пока нет завершённых результатов Парето.')
    if any(not isinstance(item, dict) or not {'alpha', 'method', 'routes', 'metrics'} <= item.keys()
           for item in networks):
        raise ValueError('Старый формат результатов: повторите расчёт Парето на Mumford 0.')
    return folder, networks


def show_pareto_routes(config, alpha, method=None):
    """Draw a saved Mumford network; this never loads a policy or repeats search."""
    from tndp.data.city import load_mumford0
    folder, networks = _saved_benchmark_pareto(config)
    candidates = [item for item in networks if isclose(item['alpha'], float(alpha))
                  and (method is None or item['method'] == method)]
    if not candidates:
        raise ValueError(f'Нет сохранённой сети для α={alpha:g}, метод={method}.')
    selected = min(candidates, key=lambda item: item['metrics']['cost'])
    city = load_mumford0(config.benchmark.path)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    plot_routes(ax, city, selected['routes'],
                f'Mumford 0 · {selected["method"]} · α = {selected["alpha"]:g}')
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_axis_off()
    name = selected['method'].replace(' ', '_')
    fig.savefig(folder / f'routes_alpha_{selected["alpha"]:g}_{name}.png', dpi=150)
    plt.show()
    return fig


def show_pareto_route_selector(config):
    """Select a saved alpha and method with notebook dropdowns."""
    import ipywidgets as widgets
    from IPython.display import display
    _, networks = _saved_benchmark_pareto(config)
    available = [float(alpha) for alpha in config.pareto_weights
                 if any(isclose(item['alpha'], float(alpha)) for item in networks)]
    if not available:
        raise ValueError('Нет сохранённых сетей для значений config.pareto_weights.')
    initial_alpha = .5 if .5 in available else available[0]

    def methods_for(alpha):
        return list(dict.fromkeys(item['method'] for item in networks
                                  if isclose(item['alpha'], alpha)))

    methods = methods_for(initial_alpha)
    initial_method = next((name for name in methods if name.startswith('NEA')), methods[0])
    alpha_choice = widgets.Dropdown(options=[(f'{a:g}', a) for a in available],
                                    value=initial_alpha, description='α:')
    method_choice = widgets.Dropdown(options=methods, value=initial_method,
                                     description='Метод:', layout=widgets.Layout(width='320px'))
    output = widgets.Output()
    updating = False

    def draw(change=None):
        nonlocal updating
        if updating:
            return
        updating = True
        try:
            methods = methods_for(alpha_choice.value)
            previous = method_choice.value
            method_choice.options = methods
            method_choice.value = previous if previous in methods else methods[0]
            output.clear_output(wait=True)
            with output:
                show_pareto_routes(config, alpha_choice.value, method_choice.value)
        finally:
            updating = False

    alpha_choice.observe(draw, names='value')
    method_choice.observe(draw, names='value')
    selector = widgets.VBox([widgets.HBox([alpha_choice, method_choice]), output])
    display(selector)
    draw()
    return selector


def show_benchmark(evaluation, output_dir):
    """Only the best benchmark network, rather than a map for every candidate."""
    from pathlib import Path
    row = evaluation['scores'].loc[evaluation['scores'].cost.idxmin()]
    result = evaluation['results'][row.method, int(row.city)]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    plot_routes(ax, evaluation['city'], result['routes'], f'Mumford 0 · {row.method}')
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_axis_off()
    fig.savefig(Path(output_dir) / 'mumford0/routes.png', dpi=150)
    plt.show()
    return fig


def _kemerovo_background(ax, data, roads=False):
    from matplotlib.collections import LineCollection
    import numpy as np
    if roads:
        segments = [list(edge['geometry'].coords) for _, _, edge in data.roads.edges(data=True)]
        segments = [np.asarray(segment) / 1000 for segment in segments]
    else:
        segments = [data.city.coords[[a, b]] for a, b in data.city.graph.edges]
    ax.add_collection(LineCollection(segments, colors='0.78', linewidths=.55, zorder=1))
    ax.scatter(data.city.coords[:, 0], data.city.coords[:, 1], s=7, color='0.4', zorder=2)
    ax.autoscale_view()
    ax.set(aspect='equal', xlabel='UTM x, км', ylabel='UTM y, км')
    ax.ticklabel_format(useOffset=False, style='plain')


def show_kemerovo_routes(data, routes, output_dir, *, projected=None):
    """Same stop IDs and route colours before and after lifting to road geometry."""
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    _kemerovo_background(ax, data, roads=projected is not None)
    colors = plt.get_cmap('tab20')
    for i, route in enumerate(routes):
        coordinates = data.city.coords[route] if projected is None else projected[i]['coordinates']
        ax.plot(coordinates[:, 0], coordinates[:, 1], color=colors(i % 20), linewidth=2,
                label=f'{i + 1}', alpha=.9, zorder=3)
    title = 'Кемерово · упрощённый граф' if projected is None else 'Кемерово · маршруты по геометрии дорог'
    ax.set_title(title)
    if routes:
        ax.legend(title='Маршруты', ncol=min(10, len(routes)), loc='upper center',
                  bbox_to_anchor=(.5, -.06), fontsize=8)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / ('routes_roads.png' if projected is not None else 'routes_simplified.png'), dpi=150)
    plt.show()
    return fig


def _add_kemerovo_basemap(ax, output_dir, zoom=12):
    """Cache only the displayed viewport; repeated plots reuse the local raster."""
    import contextily as ctx
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    raster = folder / f'basemap_osm_z{zoom}.tif'
    if not raster.is_file():
        west, east = ax.get_xlim()
        south, north = ax.get_ylim()
        ctx.bounds2raster(west, south, east, north, str(raster), zoom=zoom,
                         source=ctx.providers.OpenStreetMap.Mapnik,
                         headers={'User-Agent': 'TNDP-Teaching/0.1 (Kemerovo educational notebook)'},
                         n_connections=1, max_retries=1, timeout=(5, 20))
    ctx.add_basemap(ax, source=str(raster), attribution='© OpenStreetMap contributors',
                    attribution_size=7, reset_extent=True, alpha=.82)


def show_kemerovo_comparison(data, routes, output_dir, *, projected):
    """One figure: identical route colours and viewport, schematic vs actual roads."""
    import numpy as np
    from pyproj import Transformer
    from matplotlib.collections import LineCollection
    transformer = Transformer.from_crs(data.crs, 'EPSG:3857', always_xy=True)

    def mercator(coordinates):
        coordinates = np.asarray(coordinates)
        return np.column_stack(transformer.transform(coordinates[:, 0] * 1000,
                                                     coordinates[:, 1] * 1000))

    if len(routes) != len(projected) or any(list(route) != item['stops']
                                          for route, item in zip(routes, projected)):
        raise ValueError('Projected road routes must correspond to the same stop sequences')
    stops = mercator(data.city.coords)
    roads = [mercator(item['coordinates']) for item in projected]
    points = np.concatenate([stops, *roads])
    lower, upper = points.min(0), points.max(0)
    padding = np.maximum((upper - lower) * .04, 100)
    fig, axes = plt.subplots(1, 2, figsize=(15, 11))
    fig.subplots_adjust(left=.02, right=.98, top=.90, bottom=.11, wspace=.03)
    for ax in axes:
        ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
        ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
        ax.set_aspect('equal')
        ax.set_axis_off()
    axes[0].add_collection(LineCollection([stops[[a, b]] for a, b in data.city.graph.edges],
                                         colors='0.8', linewidths=.5, zorder=1))
    axes[0].scatter(stops[:, 0], stops[:, 1], s=6, color='0.45', zorder=2)
    _add_kemerovo_basemap(axes[1], output_dir)
    colors = plt.get_cmap('tab20')
    for i, (route, road) in enumerate(zip(routes, roads)):
        for ax, coordinates in zip(axes, (stops[route], road)):
            if ax is axes[1]:
                ax.plot(coordinates[:, 0], coordinates[:, 1], color='white', linewidth=3.8,
                        alpha=.9, zorder=3)
            ax.plot(coordinates[:, 0], coordinates[:, 1], color=colors(i % 20), linewidth=2.2,
                    label=str(i + 1), alpha=.95, zorder=4)
    axes[0].set_title('Маршруты на упрощённом графе', fontsize=15)
    axes[1].set_title('Те же маршруты по дорогам', fontsize=15)
    fig.suptitle(f'Кемерово · {len(routes)} предсказанных маршрутов', fontsize=18, y=.965)
    if routes:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Маршруты', loc='lower center',
                   bbox_to_anchor=(.5, .015), ncol=min(10, len(routes)), frameon=False)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / 'routes_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def show_kemerovo_edge_comparison(data, output_dir, *, edge=None, routes=()):
    """Zoom into one identical pair of stops on the model graph and on the roads."""
    import numpy as np
    from tndp.data.kemerovo import road_edge_coordinates
    if edge is None:
        candidates = ([(a, b) for route in routes for a, b in zip(route, route[1:])]
                      if routes else list(data.city.graph.edges))
        def detour(pair):
            a, b = pair
            direct = np.linalg.norm(data.city.coords[b] - data.city.coords[a]) * 1000
            return data.roads[a][b]['geometry'].length / max(direct, 1e-9)
        edge = max(candidates, key=detour)
    source, target = edge
    endpoints = data.city.coords[[source, target]]
    road = road_edge_coordinates(data, source, target)
    lower = np.minimum(road.min(0), endpoints.min(0))
    upper = np.maximum(road.max(0), endpoints.max(0))
    padding = np.maximum((upper - lower) * .15, .08)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, real in zip(axes, (False, True)):
        _kemerovo_background(ax, data, roads=real)
        coordinates = road if real else endpoints
        ax.plot(coordinates[:, 0], coordinates[:, 1], color='#1264a3', linewidth=3, zorder=4)
        ax.scatter(endpoints[:, 0], endpoints[:, 1], s=65, color='#d62728', zorder=5)
        for node, point in zip(edge, endpoints):
            ax.annotate(str(node), point, xytext=(7, 7), textcoords='offset points', fontweight='bold')
        ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
        ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
        ax.set_title('Упрощённое ребро' if not real else 'То же ребро по дороге')
    minutes = data.city.graph[source][target]['weight']
    fig.suptitle(f'Кемерово · ребро {source} ↔ {target} · {minutes:g} мин')
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / 'edge_simplified_vs_road.png', dpi=150)
    plt.show()
    return fig
