import math
from collections.abc import Mapping
from typing import Any


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
    anneal_epochs = max(1, anneal_timesteps // batch_size)
    progress = min(epoch / anneal_epochs, 1.0)
    minimum = learning_rate * float(config["min_lr_ratio"])
    return minimum + 0.5 * (learning_rate - minimum) * (
        1 + math.cos(math.pi * progress)
    )
