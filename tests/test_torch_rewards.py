import torch

from pufferlib.rewards import training_rewards


def test_training_rewards_preserve_environment_magnitudes() -> None:
    rewards = torch.tensor([[-4.0, -0.5, 0.0, 0.5, 7.0]])

    prepared = training_rewards(rewards)

    torch.testing.assert_close(prepared, rewards)
    assert prepared.is_contiguous()
