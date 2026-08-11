import math
from collections.abc import Mapping
from typing import Any


def rollout_epochs(total_timesteps: int, batch_size: int) -> int:
    if total_timesteps <= 0 or batch_size <= 0:
        raise ValueError("total_timesteps and batch_size must be positive")
    if total_timesteps % batch_size:
        raise ValueError("total_timesteps must be divisible by batch_size")
    return total_timesteps // batch_size


def learning_rate_at_epoch(
    config: Mapping[str, Any],
    epoch: int,
    batch_size: int,
) -> float:
    learning_rate = float(config["learning_rate"])
    if not config["anneal_lr"] or epoch <= 0:
        return learning_rate
    anneal_timesteps = int(
        config.get("lr_anneal_timesteps") or config["total_timesteps"]
    )
    anneal_epochs = rollout_epochs(anneal_timesteps, batch_size)
    progress = min(epoch / anneal_epochs, 1.0)
    minimum = learning_rate * float(config["min_lr_ratio"])
    return minimum + 0.5 * (learning_rate - minimum) * (
        1 + math.cos(math.pi * progress)
    )
