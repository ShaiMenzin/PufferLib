import sys
import types

import torch

import pufferlib

try:
    from pufferlib import _C
except ImportError:
    _C = types.ModuleType("pufferlib._C")
    _C.precision_bytes = 4
    pufferlib._C = _C
    sys.modules["pufferlib._C"] = _C

from pufferlib.torch_pufferl import PuffeRL, _torch_puff_advantage, sample_logits


class _Profile:
    def mark(self, _index: int) -> None:
        pass

    def elapsed(self, _category: int, _start: int, _end: int) -> None:
        pass


class _Vec:
    def __init__(
        self,
        observations: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
    ) -> None:
        self.observations = observations
        self.rewards = rewards
        self.terminals = terminals
        self.step = 0

    def cpu_step(self, _actions_ptr: int) -> None:
        self.step += 1
        self.observations.fill_(float(self.step))
        self.rewards.fill_(float(10 * self.step))
        self.terminals.zero_()

    def log(self) -> dict[str, float]:
        return {}


def test_rollout_rewards_align_with_actions_and_bootstrap_the_final_state() -> None:
    trainer = PuffeRL.__new__(PuffeRL)
    trainer.profile = _Profile()
    trainer.config = {"horizon": 2}
    trainer.device = torch.device("cpu")
    trainer.total_agents = 1
    trainer.gpu = False
    trainer.state = ()
    trainer.observations = torch.zeros(2, 1, 1)
    trainer.actions = torch.zeros(2, 1, 1)
    trainer.values = torch.zeros(2, 1)
    trainer.logprobs = torch.zeros(2, 1)
    trainer.rewards = torch.zeros(2, 1)
    trainer.terminals = torch.zeros(2, 1)
    trainer.episode_starts = torch.zeros(2, 1)
    trainer.global_step = 0
    trainer.vec_obs = torch.zeros(1, 1)
    trainer.vec_rewards = torch.zeros(1)
    trainer.vec_terminals = torch.zeros(1)
    trainer._vec = _Vec(
        trainer.vec_obs,
        trainer.vec_rewards,
        trainer.vec_terminals,
    )

    def forward_eval(
        observations: torch.Tensor,
        state: tuple[()],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[()]]:
        logits = torch.zeros(len(observations), 2)
        return logits, observations[:, :1], state

    trainer._forward_eval = forward_eval

    trainer.rollouts()

    torch.testing.assert_close(trainer.rewards[:, 0], torch.tensor([10.0, 20.0]))
    torch.testing.assert_close(trainer.bootstrap_values, torch.tensor([2.0]))


def test_torch_advantage_matches_transition_alignment() -> None:
    values = torch.tensor([[1.0, 2.0]])
    rewards = torch.tensor([[3.0, 4.0]])
    terminals = torch.tensor([[0.0, 1.0]])
    importance = torch.ones_like(values)
    advantages = torch.zeros_like(values)
    bootstrap_values = torch.tensor([5.0])

    result = _torch_puff_advantage(
        values,
        rewards,
        terminals,
        importance,
        advantages,
        bootstrap_values,
        0.5,
        0.5,
        1.0,
        1.0,
    )

    torch.testing.assert_close(result, torch.tensor([[3.5, 2.0]]))


def test_sample_logits_preserves_equal_multidiscrete_head_layout() -> None:
    logits = (
        torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]),
    )
    actions = torch.tensor([[2, 0], [0, 2]])

    _, logprob, entropy = sample_logits(logits, action=actions)

    stacked = torch.stack(logits, dim=0)
    normalized = stacked - stacked.logsumexp(dim=-1, keepdim=True)
    expected_logprob = normalized.gather(
        -1,
        actions.T.unsqueeze(-1),
    ).squeeze(-1).sum(dim=0)
    expected_entropy = -(normalized * normalized.exp()).sum(dim=-1).sum(dim=0)
    torch.testing.assert_close(logprob, expected_logprob)
    torch.testing.assert_close(entropy, expected_entropy)
