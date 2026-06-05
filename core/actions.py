"""The six tools exposed to the model.

Each tool takes only model-visible args and one hidden `sim` arg injected by
WanderbenchEnv.update_tool_args. The tool returns a dict — the env converts it
into an image_url content block on the way out.
"""
from __future__ import annotations

import base64
import io

from core.sim import WorldSim, Frame


def _frame_to_tool_result(frame: Frame) -> dict:
    # JPEG quality 88: ~74× faster encode than PNG and ~6× smaller wire payload, with
    # negligible visible artifacts on overlays at this resolution.
    buf = io.BytesIO()
    frame.image.save(buf, format="JPEG", quality=88, optimize=False)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "image_b64": b64,
        "image_format": "jpeg",
        "meta": frame.meta,
    }


def open_map(sim: WorldSim) -> dict:
    """Open the static reference map. Shows start pin (green), goal pin (red), and
    street labels. Does NOT show your current location. Use it to plan, then close it
    to navigate."""
    return _frame_to_tool_result(sim.open_map())


def close_map(sim: WorldSim) -> dict:
    """Close the map and return to the street-view panorama."""
    return _frame_to_tool_result(sim.close_map())


def mouse_down(sim: WorldSim) -> dict:
    """Press and hold the mouse button at the current cursor position. Subsequent
    move_cursor calls will drag (pan the view in panorama mode). Releases on mouse_up."""
    return _frame_to_tool_result(sim.mouse_down())


def mouse_up(sim: WorldSim) -> dict:
    """Release the mouse button. If the cursor barely moved since mouse_down, this
    registers as a CLICK at the current cursor position. Otherwise the drag commits
    (the pan is already applied). Clicking on the ground or on a chevron arrow in the
    panorama advances you along that bearing."""
    return _frame_to_tool_result(sim.mouse_up())


def move_cursor(direction_deg: float, distance_px: int, sim: WorldSim) -> dict:
    """Move the cursor by a vector from its CURRENT position (the cursor is persistent
    and visible in every observation as a crosshair).

    Args:
        direction_deg: angle in degrees, 0 = right, 90 = up, 180 = left, 270 = down.
        distance_px: how far to move in pixels (1 to 2000).
    """
    return _frame_to_tool_result(sim.move_cursor(direction_deg, distance_px))


def scroll_wheel(delta_y: int, sim: WorldSim) -> dict:
    """Zoom. In panorama view: positive zooms in (narrower FOV), negative zooms out.
    In map view: positive zooms map in, negative zooms out. Range: -10 to 10 per call."""
    return _frame_to_tool_result(sim.scroll_wheel(delta_y))


def submit_guess(sim: WorldSim) -> dict:
    """Declare that you've arrived at the goal. Episode ENDS immediately and your
    current position is scored against the true goal — closer is better, with a
    graduated reward from 1.0 (on the goal) to 0.0 (as far as the start was).

    You get ONE attempt. Only call this when you're confident you're at the
    destination. There is no step limit, so navigate as long as you need first."""
    return _frame_to_tool_result(sim.submit_guess())


ALL_TOOLS = [open_map, close_map, mouse_down, mouse_up, move_cursor, scroll_wheel, submit_guess]
