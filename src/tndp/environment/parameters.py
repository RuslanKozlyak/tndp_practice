"""Общие параметры среды и единиц измерения стоимости."""
from dataclasses import dataclass

@dataclass(frozen=True)
class CostConstants:
    symmetric_routes: bool
    mean_stop_time_s: float
    avg_transfer_wait_time_s: float
