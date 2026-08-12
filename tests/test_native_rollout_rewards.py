import importlib
import sys

import pytest
import torch

pytest.importorskip("torch")

try:
    _C = importlib.import_module("pufferlib._C")
except (ImportError, OSError) as exc:
    pytest.skip(f"pufferlib._C not built: {exc}", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("no CUDA device available", allow_module_level=True)

if _C.precision_bytes != 4:
    pytest.skip("native rollout inspection requires a float32 build", allow_module_level=True)


class _CudaPtr:
    def __init__(self, ptr: int, shape: tuple[int, ...]) -> None:
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": shape,
            "typestr": "<f4",
            "version": 2,
        }


def _load_args() -> dict:
    from pufferlib.pufferl import load_config

    saved = sys.argv
    sys.argv = ["pytest"]
    try:
        args = load_config(_C.env_name)
        args["cudagraphs"] = 0
        return args
    finally:
        sys.argv = saved


def _tensor(tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.as_tensor(_CudaPtr(tensor.data_ptr, shape), device="cuda")


def test_native_rollout_rewards_align_with_actions() -> None:
    if _C.env_name != "chain_mdp":
        pytest.skip("alignment fixture requires a chain_mdp build")

    args = _load_args()
    args["env"]["size"] = 2
    args["vec"].update(total_agents=64, num_buffers=2, num_threads=2)
    args["policy"].update(hidden_size=32, num_layers=1)
    args["train"].update(horizon=8, minibatch_size=64)
    trainer = _C.create_pufferl(args)
    try:
        _C.rollouts(trainer)
        horizon = int(args["train"]["horizon"])
        agents = int(args["vec"]["total_agents"])
        actions = _tensor(trainer.rollouts.actions, (horizon, agents, 1))
        rewards = _tensor(trainer.rollouts.rewards, (horizon, agents))
        expected = torch.where(actions[..., 0] == 0, 0.001, 1.0)
        torch.testing.assert_close(rewards, expected)
    finally:
        _C.close(trainer)


def test_native_training_preserves_reward_magnitudes() -> None:
    args = _load_args()
    args["train"]["replay_ratio"] = 0.0
    trainer = _C.create_pufferl(args)
    try:
        horizon = int(args["train"]["horizon"])
        agents = int(args["vec"]["total_agents"])
        rewards = _tensor(trainer.rollouts.rewards, (horizon, agents))
        rewards[0, 0] = -4.0
        rewards[1, 0] = 7.0
        torch.cuda.synchronize()

        _C.train(trainer)

        training_rewards = _tensor(
            trainer.train_rollouts.rewards,
            (agents, horizon),
        )
        torch.testing.assert_close(
            training_rewards[0, :2],
            torch.tensor([-4.0, 7.0], device="cuda"),
        )
    finally:
        _C.close(trainer)
