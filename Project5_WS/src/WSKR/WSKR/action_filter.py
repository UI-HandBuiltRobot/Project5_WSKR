"""Post-processing filter for MLP drive outputs.

Loads a params sidecar JSON (paired with a trained model) that captures the
**same** stiction, hysteresis, and saturation settings used during training +
validation, and applies them on hardware so runtime behavior matches the
simulator the model was trained against.

Sidecar schema (see params_Model_R1_Hardware_heading.json):

    saturation_limits:
        rotation_rate_dps:  max |rate| in deg/s
        drive_speed_mps:    max |drive_speed| in m/s
        vx_mps, vy_mps:     max |vx| / |vy| in m/s (strafe modes)
    stiction:
        min_turn_rate_dps:     snap-up threshold for small non-zero rates
        inner_deadband_dps:    |rate| below this is zeroed when not turning
    recommended_hysteresis:
        rotation_rate_exit_dps:  |rate| below this drops a turn (was-turning)

Rotation pipeline (dual-band + hysteresis):

    was_turning = False:
        |rate| < inner_db                -> 0
        inner_db <= |rate| < min_dps     -> sign(rate) * min_dps (snap-up)
        |rate| >= min_dps                -> rate     (was_turning := True)

    was_turning = True:
        |rate| < exit_dps                -> 0        (was_turning := False)
        |rate| >= exit_dps               -> rate

Saturation is applied last so the output never exceeds the trained limit.

Missing sidecar keys degrade gracefully to pass-through — so a model without
a paired params file behaves exactly like raw MLP output.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


def _clamp(value: float, limit: Optional[float]) -> float:
    if limit is None:
        return value
    return max(-limit, min(limit, value))


class ActionFilter:
    def __init__(
        self,
        params_path: Optional[str] = None,
        apply_stiction_hysteresis: bool = True,
        apply_saturation: bool = True,
    ) -> None:
        # Passthrough defaults
        self._rot_max_dps: Optional[float] = None
        self._drive_max_mps: Optional[float] = None
        self._vx_max_mps: Optional[float] = None
        self._vy_max_mps: Optional[float] = None
        self._rot_min_dps: float = 0.0
        self._rot_inner_db_dps: float = 0.0
        self._rot_exit_dps: float = 0.0

        # Runtime toggles. Disable either to test the raw MLP output.
        self.apply_stiction_hysteresis: bool = apply_stiction_hysteresis
        self.apply_saturation: bool = apply_saturation

        self._was_turning: bool = False
        self._loaded_from: Optional[str] = None

        if params_path is None:
            return

        path = Path(params_path)
        if not path.exists():
            return

        p = json.loads(path.read_text(encoding='utf-8'))

        sat = p.get('saturation_limits', {}) or {}
        self._rot_max_dps = _positive_or_none(sat.get('rotation_rate_dps'))
        self._drive_max_mps = _positive_or_none(sat.get('drive_speed_mps'))
        self._vx_max_mps = _positive_or_none(sat.get('vx_mps'))
        self._vy_max_mps = _positive_or_none(sat.get('vy_mps'))

        st = p.get('stiction', {}) or {}
        self._rot_min_dps = float(st.get('min_turn_rate_dps', 0.0) or 0.0)
        self._rot_inner_db_dps = float(st.get('inner_deadband_dps', 0.0) or 0.0)

        hy = p.get('recommended_hysteresis', {}) or {}
        self._rot_exit_dps = float(hy.get('rotation_rate_exit_dps', 0.0) or 0.0)
        if self._rot_exit_dps <= 0.0 and self._rot_min_dps > 0.0:
            # Sensible fallback matching typical exit ≈ 0.6 * min_turn.
            self._rot_exit_dps = 0.6 * self._rot_min_dps

        self._loaded_from = str(path)

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Reset hysteresis state. Call at episode start."""
        self._was_turning = False

    @property
    def loaded_from(self) -> Optional[str]:
        return self._loaded_from

    @property
    def has_stiction(self) -> bool:
        return self._rot_min_dps > 0.0

    # ------------------------------------------------------------------ apply

    def process_rotation_dps(self, raw_dps: float) -> float:
        """Apply dual-band stiction + hysteresis, then saturation. Each
        stage is skipped individually if its toggle is False."""
        rate = self._stiction_hysteresis(raw_dps) if self.apply_stiction_hysteresis else raw_dps
        if self.apply_saturation:
            rate = _clamp(rate, self._rot_max_dps)
        return rate

    def clamp_rotation_dps(self, value: float) -> float:
        return _clamp(value, self._rot_max_dps) if self.apply_saturation else value

    def clamp_drive_mps(self, value: float) -> float:
        return _clamp(value, self._drive_max_mps) if self.apply_saturation else value

    def clamp_vx_mps(self, value: float) -> float:
        return _clamp(value, self._vx_max_mps) if self.apply_saturation else value

    def clamp_vy_mps(self, value: float) -> float:
        return _clamp(value, self._vy_max_mps) if self.apply_saturation else value

    # ------------------------------------------------------------------ inner

    def _stiction_hysteresis(self, rate_dps: float) -> float:
        if self._rot_min_dps <= 0.0:
            return rate_dps

        mag = abs(rate_dps)

        if self._was_turning:
            if mag < self._rot_exit_dps:
                self._was_turning = False
                return 0.0
            return rate_dps

        # not turning
        if mag < self._rot_inner_db_dps or mag == 0.0:
            return 0.0

        self._was_turning = True
        if mag < self._rot_min_dps:
            return math.copysign(self._rot_min_dps, rate_dps)
        return rate_dps


def _positive_or_none(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0.0 else None
