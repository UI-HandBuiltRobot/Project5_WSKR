"""MLP policy runtime: JSON weight loader + history buffer.

Exposes two classes:

- ``JsonMLPModel``: loads weights, biases, and input/output normalization
  statistics from a single JSON file; runs a forward pass in NumPy and
  returns outputs in physical units.

- ``TemporalInputBuilder``: assembles the model's feature vector from a
  rolling window of sensory readings (11 whisker ranges in metres +
  1 heading-to-target in degrees) and past drive commands.

Feature layout (oldest first, memory_steps = N):
    for i in 0..N-1:
        whisker_lengths[i]     # 11 floats
        heading_to_target[i]   # 1 float
        if i < N-1:
            action_slice[i]    # physical units

Padding on episode start copies the first real observation (NOT zeros)
so whisker readings stay in-distribution. Action-slot pads are zero.

Call ``TemporalInputBuilder.reset()`` at the start of every new episode
so history from a previous approach doesn't leak in.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np


LEAKY_ALPHA: float = 0.01
_STD_FLOOR: float = 1e-6


class JsonMLPModel:
    """Pure-NumPy MLP loaded from a JSON weight blob."""

    def __init__(self, model_path: str) -> None:
        data = json.loads(Path(model_path).read_text(encoding='utf-8'))

        self.activation: str = data.get('activation', 'relu')
        self.memory_steps: int = int(data.get('memory_steps', 1))
        self.input_dim: int = int(data.get('input_dim', 0))
        self.output_dim: int = int(data.get('output_dim', 0))
        # Accept both the old 'heading' tag and the canonical 'heading_drive'.
        self.model_mode: str = data.get('model_mode', 'heading_drive')

        self.weights: List[np.ndarray] = [
            np.asarray(w, dtype=np.float64) for w in data['weights']
        ]
        self.biases: List[np.ndarray] = [
            np.asarray(b[0], dtype=np.float64) if isinstance(b[0], list)
            else np.asarray(b, dtype=np.float64)
            for b in data['biases']
        ]

        self.x_mean = np.asarray(data['x_mean'][0], dtype=np.float64)
        self.x_std = np.asarray(data['x_std'][0], dtype=np.float64)
        self.y_mean = np.asarray(data['y_mean'][0], dtype=np.float64)
        self.y_std = np.asarray(data['y_std'][0], dtype=np.float64)

        if self.x_mean.shape[0] != self.input_dim:
            raise ValueError(
                f'x_mean length {self.x_mean.shape[0]} != input_dim {self.input_dim}'
            )
        if self.y_mean.shape[0] != self.output_dim:
            raise ValueError(
                f'y_mean length {self.y_mean.shape[0]} != output_dim {self.output_dim}'
            )

        # Floor near-zero stds to avoid divide-by-zero during normalization.
        self.x_std = np.where(np.abs(self.x_std) < _STD_FLOOR, 1.0, self.x_std)
        self.y_std = np.where(np.abs(self.y_std) < _STD_FLOOR, 1.0, self.y_std)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return np.maximum(0.0, x)
        if self.activation == 'tanh':
            return np.tanh(x)
        if self.activation == 'leaky_relu':
            return np.where(x > 0, x, LEAKY_ALPHA * x)
        raise ValueError(f'Unsupported activation: {self.activation!r}')

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. Input and output are in physical units; normalization
        stats are applied internally. Returns a 1-D ndarray of length output_dim."""
        if x.shape[0] != self.input_dim:
            raise ValueError(f'Input length {x.shape[0]} != input_dim {self.input_dim}')
        a = (x - self.x_mean) / self.x_std
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = self._activate(a @ w + b)
        z = a @ self.weights[-1] + self.biases[-1]
        return (z * self.y_std + self.y_mean).flatten()


def _action_dim_from_mode(model_mode: str, fallback_dim: int) -> int:
    if model_mode in ('heading', 'heading_drive', 'xy_strafe'):
        return 2
    if model_mode == 'heading_strafe':
        return 3
    return fallback_dim


class TemporalInputBuilder:
    """Rolling sensory + action history window, shaped for the MLP input."""

    SENSORY_DIM: int = 12  # 11 whiskers + 1 heading

    def __init__(self, model: JsonMLPModel) -> None:
        self.model = model
        self.n: int = model.memory_steps
        self.action_dim: int = _action_dim_from_mode(model.model_mode, model.output_dim)
        self.sensory_history: List[np.ndarray] = []
        self.action_history: List[np.ndarray] = []

    def reset(self) -> None:
        """Clear history. Call at every episode start or policy pause/resume."""
        self.sensory_history.clear()
        self.action_history.clear()

    def push_sensory(self, whiskers: np.ndarray, heading_deg: float) -> None:
        """Append one sensory timestep (whiskers in metres, heading in degrees)."""
        sensory = np.concatenate([
            whiskers.astype(np.float64),
            np.array([heading_deg], dtype=np.float64),
        ])
        self.sensory_history.append(sensory)
        if len(self.sensory_history) > self.n:
            self.sensory_history.pop(0)

    def push_action(self, action: np.ndarray) -> None:
        """Append the command just issued, so the next tick can condition on it."""
        self.action_history.append(action.astype(np.float64))
        if len(self.action_history) > max(0, self.n - 1):
            self.action_history.pop(0)

    def build_input(self) -> np.ndarray:
        """Assemble the feature vector for the current tick.

        Copy-pads from the first real observation when history is short so
        whisker readings stay in-distribution. Action-slot pads are zero.
        """
        if not self.sensory_history:
            raise ValueError('No sensory data available yet; call push_sensory first.')

        pad_sensory = self.sensory_history[0].copy()
        pad_action = np.zeros((self.action_dim,), dtype=np.float64)

        sens_pad = [pad_sensory] * (self.n - len(self.sensory_history))
        act_pad = [pad_action] * ((self.n - 1) - len(self.action_history))
        padded_sensory = sens_pad + list(self.sensory_history)
        padded_actions = act_pad + list(self.action_history)

        chunks: List[np.ndarray] = []
        for i in range(self.n):
            chunks.append(padded_sensory[i])
            if i < self.n - 1:
                chunks.append(padded_actions[i])

        x = np.concatenate(chunks)
        if x.shape[0] != self.model.input_dim:
            raise ValueError(
                f'Built input length {x.shape[0]} != model input_dim {self.model.input_dim}'
            )
        return x
