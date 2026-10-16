"""Behavior-cloning policy. Inference receives observations, never expert phases."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, observation_size: int = 26, width: int = 128):
        super().__init__()
        self.width = width
        self.register_buffer("mean", torch.zeros(observation_size))
        self.register_buffer("scale", torch.ones(observation_size))
        self.network = nn.Sequential(
            nn.Linear(observation_size, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 4),
        )

    def forward(self, observations):
        x = (observations - self.mean) / self.scale
        outputs = self.network(x)
        return torch.tanh(outputs[:, :3]), outputs[:, 3]

    @torch.inference_mode()
    def act(self, observation: np.ndarray) -> np.ndarray:
        movement, gripper = self(torch.from_numpy(observation).unsqueeze(0))
        return np.r_[movement[0].numpy(), 1.0 if gripper[0].item() >= 0 else -1.0].astype(
            np.float32
        )

    @classmethod
    def load(cls, path: Path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(width=checkpoint["width"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
