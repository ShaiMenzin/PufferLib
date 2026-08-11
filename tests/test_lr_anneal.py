import pytest
from pufferlib.schedules import learning_rate_at_epoch, rollout_epochs


def test_rollout_epochs_reject_a_partial_batch() -> None:
    with pytest.raises(ValueError, match="divisible"):
        rollout_epochs(total_timesteps=129, batch_size=128)


def test_lr_annealing_uses_an_absolute_timestep_horizon() -> None:
    batch_size = 128
    common = {
        "learning_rate": 3e-4,
        "min_lr_ratio": 0.1,
        "anneal_lr": True,
        "lr_anneal_timesteps": 96 * batch_size,
    }
    sweep = {**common, "total_timesteps": 40 * batch_size}
    production = {**common, "total_timesteps": 96 * batch_size}

    assert learning_rate_at_epoch(sweep, 39, batch_size) == learning_rate_at_epoch(
        production,
        39,
        batch_size,
    )
