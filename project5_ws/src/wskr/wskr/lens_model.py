"""Analytic fisheye lens model for the WSKR camera — normalized coordinates.

Everything in this module works in **width-normalized image coordinates**.
For an image of width W pixels, a pixel at (u_px, v_px) maps to:
    u_norm = u_px / W
    v_norm = v_px / W   (intentionally normalized by W, not H, so isotropic)

This keeps the model resolution-agnostic: the same LensParams work whether
the downstream image is 1920x1080 or 640x360, as long as aspect ratio is
preserved. Callers scale normalized coords back to pixels by multiplying
by the target image width.

Convention: + heading = LEFT, 0 = straight ahead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass
class LensParams:
    """Fisheye lens calibration in width-normalized coordinates.

    Defaults come from the reference calibration (X_MIN=176, X_MAX=1744,
    CY=540 at a 1920-wide image), divided by 1920.
    """
    x_min: float = 176.0 / 1920.0   # ~0.0917
    x_max: float = 1744.0 / 1920.0  # ~0.9083
    cy: float = 540.0 / 1920.0      # 0.28125
    hfov_deg: float = 180.0
    tilt_deg: float = 30.0
    y_offset: float = 0.0           # normalized (by width)

    @property
    def cx(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def r_max(self) -> float:
        return (self.x_max - self.x_min) / 2.0


def compute_heading_rad(u_norm: float, v_norm: float, params: LensParams) -> float:
    """Forward projection: normalized pixel -> heading in radians.

    Convention: +heading = LEFT, 0 = straight ahead.
    """
    hfov_rad = math.radians(params.hfov_deg)
    tilt_rad = math.radians(params.tilt_deg)
    r_max = params.r_max
    if r_max <= 0.0:
        return 0.0
    k = (hfov_rad / 2.0) / r_max

    x = u_norm - params.cx
    y = (v_norm - params.cy) - params.y_offset

    r = math.sqrt(x * x + y * y)
    if r < 1e-12:
        return 0.0

    theta = k * r
    dx = math.sin(theta) * (x / r)
    dy = math.sin(theta) * (y / r)
    dz = math.cos(theta)

    # Un-tilt: inverse of the R(+tilt) rotation used by project_meridian_normalized.
    # World dz from camera-frame (dy, dz) is  -dy*sin(t) + dz*cos(t).
    # The sign on the dy term was flipped previously, which caused heading to
    # drift wildly along a single meridian instead of staying constant.
    dz_world = -dy * math.sin(tilt_rad) + dz * math.cos(tilt_rad)
    return math.atan2(-dx, dz_world)


def _project_direction_norm(
    dx: float, dy: float, dz: float, params: LensParams
) -> tuple[float, float] | None:
    """Inverse projection of a unit direction in the (tilted) camera frame to
    a normalized pixel coordinate."""
    dz = max(-1.0, min(1.0, dz))
    theta = math.acos(dz)
    r = params.r_max * (theta / (math.pi / 2.0))

    norm = math.sqrt(dx * dx + dy * dy)
    if norm < 1e-9:
        return params.cx, params.cy + params.y_offset

    x = r * dx / norm
    y = r * dy / norm
    return params.cx + x, params.cy + y + params.y_offset


def project_meridian_normalized(
    heading_deg: float,
    params: LensParams,
    aspect: float = 9.0 / 16.0,
    phi_range_deg: Iterable[int] = range(-80, 81, 2),
) -> list[tuple[float, float]]:
    """Inverse projection of a heading meridian (at azimuth=heading_deg,
    varying elevation phi) to a polyline in normalized image coords.

    Points are clipped to [0, 1] in u and [0, aspect] in v (a normalized
    v_max = H/W), so the caller can blindly scale by the target image width.
    """
    psi = math.radians(heading_deg)
    tilt_rad = math.radians(params.tilt_deg)
    cos_t = math.cos(tilt_rad)
    sin_t = math.sin(tilt_rad)

    v_max = max(aspect, 0.0)

    pts: list[tuple[float, float]] = []
    for phi_deg in phi_range_deg:
        phi = math.radians(phi_deg)
        dx = math.sin(-psi) * math.cos(phi)
        dy = math.sin(phi)
        dz = math.cos(-psi) * math.cos(phi)

        dy_t = dy * cos_t - dz * sin_t
        dz_t = dy * sin_t + dz * cos_t

        proj = _project_direction_norm(dx, dy_t, dz_t, params)
        if proj is None:
            continue
        u, v = proj
        if 0.0 <= u <= 1.0 and 0.0 <= v <= v_max:
            pts.append((u, v))
    return pts
