import torch


def training_rewards(rewards: torch.Tensor) -> torch.Tensor:
    return rewards.contiguous()
