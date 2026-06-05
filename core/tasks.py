from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Task:
    task_id: str
    city: str
    start_pano_id: str
    start_lat: float
    start_lng: float
    goal_lat: float
    goal_lng: float
    optimal_steps: int
    optimal_distance_m: float
    reference_map_path: str
    world_graph_path: str = ""
    goal_radius_m: float = 25.0
    info: dict = field(default_factory=dict)

    @property
    def initial_distance_m(self) -> float:
        return _haversine_m(self.start_lat, self.start_lng, self.goal_lat, self.goal_lng)


def load_tasks(path: str | Path) -> list[Task]:
    path = Path(path)
    out: list[Task] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(Task(**row))
    return out


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    from math import radians, sin, cos, asin, sqrt
    r = 6371000.0
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return 2 * r * asin(sqrt(a))
