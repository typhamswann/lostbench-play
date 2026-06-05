"""World graph — panos and reachability links.

Used for both synthetic worlds (in `data/world_graphs/synthetic_*.jsonl`) and real
Mapillary-baked worlds. Same format, same loader.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PanoNode:
    pano_id: str
    # `lat`/`lng` is the topology position — the OSM road-centerline waypoint
    # this node represents. Used for graph structure: neighbor edges, intersection
    # bridges, BFS/shortest-path, way clipping.
    lat: float
    lng: float
    # World bearing (degrees, 0=N) corresponding to the IMAGE's x=0 — i.e. the
    # direction the camera was facing when the pano was captured. The render
    # pipeline projects the equirectangular image such that yaw=0 shows the
    # compass_angle direction; this MUST match the image's true orientation
    # or the user sees one direction while the engine believes another.
    compass_angle: float = 0.0
    neighbors: list[str] = field(default_factory=list)
    # When a node-id differs from its backing image (skeleton model: 1 node per
    # road waypoint; the same Mapillary pano may back multiple road waypoints at
    # an intersection), `image_id` is the Mapillary pano id used for imagery
    # lookup. When empty, pano_id is also the image id (dense/corridor models).
    image_id: str = ""
    # OSM road-polyline bearing at this waypoint (skeleton model only). Used by
    # post-click auto-orient to snap the camera to the road axis. Distinct from
    # compass_angle because the chosen Mapillary image may be from a sequence
    # whose car was driving perpendicular or opposite to the road's polyline
    # direction. When unset, road-snap falls back to compass_angle.
    road_bearing: float = 0.0
    # `image_lat`/`image_lng` = where the camera actually was when the photo
    # was taken (Mapillary's reported pano coordinates). Cars drive in lanes, so
    # this is usually 2-5m off the centerline. Click ray-casting and distance
    # math should use these so the engine's "current position" matches what
    # the user sees in the panorama. When unset (legacy bakes), fall back to
    # the centerline lat/lng.
    image_lat: float = 0.0
    image_lng: float = 0.0

    @property
    def cam_lat(self) -> float:
        """The camera's true position — image_lat if set, else centerline lat."""
        return self.image_lat if self.image_lat != 0.0 else self.lat

    @property
    def cam_lng(self) -> float:
        return self.image_lng if self.image_lng != 0.0 else self.lng


@dataclass
class WorldGraph:
    panos: dict[str, PanoNode]

    def get(self, pano_id: str) -> PanoNode:
        return self.panos[pano_id]

    def neighbors_of(self, pano_id: str) -> list[PanoNode]:
        return [self.panos[n] for n in self.panos[pano_id].neighbors if n in self.panos]

    def __contains__(self, pano_id: str) -> bool:
        return pano_id in self.panos

    def __len__(self) -> int:
        return len(self.panos)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "WorldGraph":
        path = Path(path)
        panos: dict[str, PanoNode] = {}
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # Filter to fields PanoNode knows about — newer bakes may add
                # auxiliary fields like corridor_id that we don't use at runtime.
                allowed = {"pano_id", "lat", "lng", "compass_angle", "neighbors",
                           "image_id", "road_bearing", "image_lat", "image_lng"}
                node = PanoNode(**{k: v for k, v in row.items() if k in allowed})
                panos[node.pano_id] = node
        return cls(panos=panos)


def bearing_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle initial bearing from (lat1, lng1) to (lat2, lng2). 0=N, 90=E."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlng = math.radians(lng2 - lng1)
    y = math.sin(dlng) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlng)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angular_diff_deg(a: float, b: float) -> float:
    """Signed shortest-arc difference (a - b) in degrees, in [-180, 180]."""
    d = (a - b + 540.0) % 360.0 - 180.0
    return d


def move_along_bearing(lat: float, lng: float, bearing_deg_: float, dist_m: float) -> tuple[float, float]:
    """Project (lat, lng) along an initial bearing for dist_m meters on the sphere.
    Used by click-to-go to compute the world location the user clicked on."""
    R = 6371000.0
    brg = math.radians(bearing_deg_)
    phi1 = math.radians(lat)
    lam1 = math.radians(lng)
    d_r = dist_m / R
    phi2 = math.asin(math.sin(phi1) * math.cos(d_r)
                     + math.cos(phi1) * math.sin(d_r) * math.cos(brg))
    lam2 = lam1 + math.atan2(math.sin(brg) * math.sin(d_r) * math.cos(phi1),
                              math.cos(d_r) - math.sin(phi1) * math.sin(phi2))
    return (math.degrees(phi2), math.degrees(lam2))
