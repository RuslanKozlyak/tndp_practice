"""Выбор колонок и усреднение одиночных результатов по городам."""
import pandas as pd
from tndp.config import DEFAULT_METRICS

METRICS = DEFAULT_METRICS

def compact_table(evaluation, metrics=None):
    """Average one result per method and city, without exposing run bookkeeping."""
    frame = evaluation['scores']
    names = tuple('cost' if name == 'C' else name for name in
                  (metrics if metrics is not None else evaluation.get('metrics', DEFAULT_METRICS)))
    if not names or len(set(names)) != len(names):
        raise ValueError('Choose distinct metrics')
    if frame.empty:
        return pd.DataFrame(columns=['C' if name == 'cost' else name for name in names]).rename_axis('Метод')
    missing = set(names) - set(frame.select_dtypes(include='number').columns)
    if missing:
        raise ValueError(f'Unknown or nonnumeric metrics: {sorted(missing)}. Available: {list(frame.select_dtypes(include="number").columns)}')
    if frame.duplicated(['method', 'city']).any():
        raise ValueError('Expected one result per method and city')
    return frame.groupby('method', sort=False)[list(names)].mean().rename(columns={'cost': 'C'}).rename_axis('Метод').round(3)
