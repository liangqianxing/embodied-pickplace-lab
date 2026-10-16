"""Matched-capacity baseline and explicitly translation-invariant input ablation."""

from pathlib import Path

import torch

from .policy import Policy
from .robustness import ABSOLUTE_XY


class StudyPolicy(Policy):
    def __init__(self, *, relative: bool = False, width: int = 128):
        super().__init__(width=width)
        self.relative = relative

    def features(self, observations):
        if not self.relative:
            return observations
        result = observations.clone()
        result[:, list(ABSOLUTE_XY)] = 0
        return result

    def forward(self, observations):
        return super().forward(self.features(observations))

    @classmethod
    def load(cls, path: Path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(relative=checkpoint["relative"], width=checkpoint["width"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
