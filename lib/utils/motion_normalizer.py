"""Feature-wise normalization for hybrid Gaze2HOI motion targets."""

from dataclasses import dataclass

import torch


class MaskedFeatureStats:
    """Accumulate per-feature mean/std over valid (batch, time) entries."""

    def __init__(self, feature_dim):
        self.feature_dim = int(feature_dim)
        self.count = 0
        self.sum = torch.zeros(self.feature_dim, dtype=torch.float64)
        self.square_sum = torch.zeros(self.feature_dim, dtype=torch.float64)

    def update(self, values, valid_mask):
        if values.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_dim}, got {values.shape[-1]}"
            )
        valid = valid_mask.to(device=values.device, dtype=torch.bool)
        selected = values[valid]
        if selected.numel() == 0:
            return
        selected = selected.detach().to(device="cpu", dtype=torch.float64)
        self.count += int(selected.shape[0])
        self.sum += selected.sum(dim=0)
        self.square_sum += selected.square().sum(dim=0)

    def finalize(self, min_std=1e-4):
        if self.count <= 0:
            raise RuntimeError("Cannot finalize empty motion statistics")
        mean = self.sum / self.count
        variance = self.square_sum / self.count - mean.square()
        std = variance.clamp_min(0.0).sqrt().clamp_min(float(min_std))
        return mean.float(), std.float()


@dataclass
class MotionNormalizer:
    left_mean: torch.Tensor
    left_std: torch.Tensor
    right_mean: torch.Tensor
    right_std: torch.Tensor
    object_mean: torch.Tensor
    object_std: torch.Tensor
    eps: float = 1e-6

    def _match(self, value, statistic):
        return statistic.to(device=value.device, dtype=value.dtype).view(
            *([1] * (value.dim() - 1)), -1
        )

    def _normalize_one(self, value, mean, std):
        return (value - self._match(value, mean)) / self._match(value, std).clamp_min(
            self.eps
        )

    def _denormalize_one(self, value, mean, std):
        return value * self._match(value, std) + self._match(value, mean)

    def normalize(self, left, right, obj):
        return (
            self._normalize_one(left, self.left_mean, self.left_std),
            self._normalize_one(right, self.right_mean, self.right_std),
            self._normalize_one(obj, self.object_mean, self.object_std),
        )

    def denormalize(self, left, right, obj):
        return (
            self._denormalize_one(left, self.left_mean, self.left_std),
            self._denormalize_one(right, self.right_mean, self.right_std),
            self._denormalize_one(obj, self.object_mean, self.object_std),
        )

    def state_dict(self):
        return {
            "version": 1,
            "left_mean": self.left_mean.detach().cpu(),
            "left_std": self.left_std.detach().cpu(),
            "right_mean": self.right_mean.detach().cpu(),
            "right_std": self.right_std.detach().cpu(),
            "object_mean": self.object_mean.detach().cpu(),
            "object_std": self.object_std.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state):
        required = (
            "left_mean",
            "left_std",
            "right_mean",
            "right_std",
            "object_mean",
            "object_std",
        )
        missing = [key for key in required if key not in state]
        if missing:
            raise KeyError(f"Motion normalization state is missing {missing}")
        return cls(**{key: torch.as_tensor(state[key]).float() for key in required})


def finalize_motion_normalizer(left_stats, right_stats, object_stats, min_std=1e-4):
    left_mean, left_std = left_stats.finalize(min_std=min_std)
    right_mean, right_std = right_stats.finalize(min_std=min_std)
    object_mean, object_std = object_stats.finalize(min_std=min_std)
    return MotionNormalizer(
        left_mean=left_mean,
        left_std=left_std,
        right_mean=right_mean,
        right_std=right_std,
        object_mean=object_mean,
        object_std=object_std,
    )
