import math
from collections import OrderedDict
from contextlib import contextmanager

import torch


def warmup_cosine_factor(step, warmup_steps, total_steps, min_lr_ratio=0.0):
    """Return an LR multiplier for linear warmup followed by cosine decay."""
    step = int(step)
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    min_lr_ratio = float(min_lr_ratio)
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(
        1.0,
        max(0.0, float(step - warmup_steps) / float(decay_steps)),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


class ExponentialMovingAverage:
    """EMA copy of a module state, including floating-point buffers."""

    def __init__(self, module, decay=0.9999):
        decay = float(decay)
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be between 0 and 1, got {decay}")
        self.decay = decay
        self.num_updates = 0
        self.shadow = OrderedDict(
            (name, value.detach().clone())
            for name, value in module.state_dict().items()
        )

    @torch.no_grad()
    def update(self, module):
        current_state = module.state_dict()
        if current_state.keys() != self.shadow.keys():
            raise RuntimeError("EMA state keys no longer match the model state.")
        self.num_updates += 1
        for name, current_value in current_state.items():
            shadow_value = self.shadow[name]
            current_value = current_value.detach()
            if torch.is_floating_point(shadow_value):
                shadow_value.mul_(self.decay).add_(
                    current_value.to(dtype=shadow_value.dtype),
                    alpha=1.0 - self.decay,
                )
            else:
                shadow_value.copy_(current_value)

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state):
        self.decay = float(state.get("decay", self.decay))
        self.num_updates = int(state.get("num_updates", 0))
        shadow = state.get("shadow", state)
        if shadow.keys() != self.shadow.keys():
            missing = sorted(set(self.shadow) - set(shadow))
            unexpected = sorted(set(shadow) - set(self.shadow))
            raise RuntimeError(
                "EMA checkpoint does not match the model. "
                f"Missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        for name, value in shadow.items():
            self.shadow[name].copy_(
                value.to(
                    device=self.shadow[name].device,
                    dtype=self.shadow[name].dtype,
                )
            )

    def model_state_dict(self):
        return OrderedDict(
            (name, value.detach().clone()) for name, value in self.shadow.items()
        )

    @contextmanager
    def average_parameters(self, module):
        original_state = OrderedDict(
            (name, value.detach().clone())
            for name, value in module.state_dict().items()
        )
        module.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            module.load_state_dict(original_state, strict=True)
