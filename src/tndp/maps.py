"""Paired Folium maps of simplified routes and their saved road geometries."""
from pathlib import Path

import numpy as np
from pyproj import Transformer

def _latlon(data, coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    transformer = Transformer.from_crs(data.crs, 'EPSG:4326', always_xy=True)
    lon, lat = transformer.transform(coordinates[:, 0] * 1000, coordinates[:, 1] * 1000)
    return np.column_stack((lat, lon)).tolist()


def _paired_map(bounds, titles):
    import folium
    from folium.plugins import DualMap
    from branca.element import MacroElement, Template
    center = np.asarray(bounds).mean(0).tolist()
    maps = DualMap(location=center, tiles=None, zoom_start=12,
                   control_scale=True, scroll_wheel_zoom=True)
    maps.get_root().height = '650px'
    maps.fit_bounds(bounds, padding=(20, 20))
    basemaps = []
    for index, (provider, name) in enumerate((
        ('OpenStreetMap', 'OpenStreetMap'),
        ('Esri.WorldStreetMap', 'Улицы · Esri'),
        ('Esri.WorldImagery', 'Спутник · Esri'),
    )):
        basemaps.append(tuple(folium.TileLayer(provider, name=name, show=index == 0).add_to(panel)
                              for panel in (maps.m1, maps.m2)))
    for panel, title in zip((maps.m1, maps.m2), titles):
        heading = MacroElement()
        heading.panel = panel
        heading.title = title
        heading._template = Template('''{% macro script(this, kwargs) %}
        (function () {
          const heading = L.control({position: 'bottomleft'});
          heading.onAdd = function () {
            const label = L.DomUtil.create('div');
            label.style.cssText = 'background:white;padding:6px 10px;font:600 14px sans-serif;box-shadow:0 1px 5px #777';
            label.textContent = {{ this.title | tojson }};
            return label;
          };
          heading.addTo({{ this.panel.get_name() }});
        })();
        {% endmacro %}''')
        maps.get_root().add_child(heading)
    return maps, basemaps


def _sync_layers(maps, pairs, colors, basemaps):
    from branca.element import MacroElement, Template
    sync = MacroElement()
    sync.left, sync.right, sync.pairs = maps.m1, maps.m2, pairs
    sync.colors, sync.basemaps = colors, basemaps
    sync._template = Template('''{% macro script(this, kwargs) %}
    (function () {
      const left = {{ this.left.get_name() }}, right = {{ this.right.get_name() }};
      const pairs = [
        {% for first, second in this.pairs %}
          [{{ first.get_name() }}, {{ second.get_name() }}],
        {% endfor %}
      ];
      const basemaps = [
        {% for first, second in this.basemaps %}
          [{{ first.get_name() }}, {{ second.get_name() }}],
        {% endfor %}
      ];
      const colors = {{ this.colors | tojson }};
      function paintControls(panel) {
        panel.getContainer().querySelectorAll('.leaflet-control-layers-overlays label').forEach(label => {
          const color = colors[label.textContent.trim()];
          if (!color) return;
          const checkbox = label.querySelector('input');
          checkbox.style.accentColor = color;
          let swatch = label.querySelector('.tndp-route-swatch');
          if (!swatch) {
            swatch = document.createElement('span');
            swatch.className = 'tndp-route-swatch';
            checkbox.insertAdjacentElement('afterend', swatch);
          }
          swatch.style.cssText = 'display:inline-block;width:16px;height:4px;margin:0 6px;vertical-align:middle';
          swatch.style.backgroundColor = color;
          swatch.setAttribute('aria-hidden', 'true');
        });
      }
      [left, right].forEach(paintControls);
      function mirror(event, source, add) {
        const pair = pairs.find(item => item[source] === event.layer);
        if (!pair) return;
        const target = source === 0 ? right : left, layer = pair[1-source];
        if (add && !target.hasLayer(layer)) target.addLayer(layer);
        if (!add && target.hasLayer(layer)) target.removeLayer(layer);
        [left, right].forEach(paintControls);
      }
      left.on('overlayadd', event => mirror(event, 0, true));
      left.on('overlayremove', event => mirror(event, 0, false));
      right.on('overlayadd', event => mirror(event, 1, true));
      right.on('overlayremove', event => mirror(event, 1, false));
      function mirrorBasemap(event, source) {
        const pair = basemaps.find(item => item[source] === event.layer);
        if (!pair) return;
        const target = source === 0 ? right : left, layer = pair[1-source];
        if (target.hasLayer(layer)) return;
        basemaps.forEach(item => target.removeLayer(item[1-source]));
        target.addLayer(layer);
        [left, right].forEach(paintControls);
      }
      left.on('baselayerchange', event => mirrorBasemap(event, 0));
      right.on('baselayerchange', event => mirrorBasemap(event, 1));
    })();
    {% endmacro %}''')
    maps.get_root().add_child(sync)


def kemerovo_route_map(data, routes, output_dir, *, projected):
    """Paired maps with synchronized navigation, basemaps and colored route controls."""
    import folium
    from matplotlib import colormaps
    from matplotlib.colors import to_hex
    if len(routes) != len(projected) or any(list(route) != item['stops']
                                          for route, item in zip(routes, projected)):
        raise ValueError('Projected routes must have the same stop sequences')
    stops = _latlon(data, data.city.coords)
    roads = [_latlon(data, item['coordinates']) for item in projected]
    points = np.concatenate([np.asarray(stops), *[np.asarray(road) for road in roads]])
    maps, basemaps = _paired_map([points.min(0).tolist(), points.max(0).tolist()],
                                ('Упрощённый граф', 'Те же маршруты по дорогам'))
    folium.PolyLine([[stops[a], stops[b]] for a, b in data.city.graph.edges],
                    color='#777777', weight=.8, opacity=.35, interactive=False).add_to(maps.m1)
    pairs, colors = [], {}
    for index, (route, road) in enumerate(zip(routes, roads)):
        color = to_hex(colormaps['tab20'](index % 20))
        colors[f'Маршрут {index + 1}'] = color
        groups = []
        for panel, coordinates in zip((maps.m1, maps.m2), ([stops[stop] for stop in route], road)):
            group = folium.FeatureGroup(name=f'Маршрут {index + 1}', show=True).add_to(panel)
            folium.PolyLine(coordinates, color=color, weight=4, opacity=.95,
                            tooltip=f'Маршрут {index + 1} · {len(route)} остановок').add_to(group)
            for label, node in (('Начало', route[0]), ('Конец', route[-1])):
                folium.CircleMarker(stops[node], radius=4, color=color, fill=True, fill_opacity=1,
                                    tooltip=f'{label} маршрута {index + 1} · узел {node}').add_to(group)
            groups.append(group)
        pairs.append(tuple(groups))
    stop_groups = []
    for panel in (maps.m1, maps.m2):
        group = folium.FeatureGroup(name='Остановки', show=True).add_to(panel)
        for node, location in enumerate(stops):
            serving = [str(index + 1) for index, route in enumerate(routes) if node in route]
            detail = 'Маршруты: ' + ', '.join(serving) if serving else 'Не обслуживается'
            folium.CircleMarker(location, radius=3, color='#333333', weight=1,
                                fill=True, fill_color='white', fill_opacity=1,
                                tooltip=f'Остановка {node}<br>{detail}').add_to(group)
        stop_groups.append(group)
    pairs.append(tuple(stop_groups))
    folium.LayerControl(collapsed=False).add_to(maps.m2)
    _sync_layers(maps, pairs, colors, basemaps)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    maps.save(str(folder / 'routes_comparison.html'))
    return maps
