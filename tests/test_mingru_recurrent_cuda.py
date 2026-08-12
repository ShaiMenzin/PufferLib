import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or shutil.which("nvcc") is None,
    reason="CUDA and nvcc are required",
)


def _pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _reference(
    combined: torch.Tensor,
    state: torch.Tensor,
    inputs: torch.Tensor,
    episode_starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = []
    recurrent_state = state
    for step in range(combined.shape[1]):
        recurrent_state = recurrent_state * (~episode_starts[:, step]).unsqueeze(1)
        hidden, gate, projection = combined[:, step].chunk(3, dim=1)
        hidden = torch.where(hidden >= 0, hidden + 0.5, hidden.sigmoid())
        recurrent_state = torch.lerp(recurrent_state, hidden, gate.sigmoid())
        highway = projection.sigmoid()
        outputs.append(highway * recurrent_state + (1.0 - highway) * inputs[:, step])
    return torch.stack(outputs, dim=1), recurrent_state.unsqueeze(1)


def test_native_mingru_respects_initial_state_and_episode_boundaries(
    tmp_path: Path,
) -> None:
    source = Path(__file__).with_suffix(".cu")
    library_path = tmp_path / "mingru_recurrent.so"
    subprocess.run(
        [
            "nvcc",
            "-shared",
            "-o",
            str(library_path),
            str(source),
            "-lcublas",
            "-lcurand",
            "--compiler-options",
            "-fPIC",
            "-Xcompiler",
            "-O2",
            "-arch=native",
        ],
        check=True,
    )
    library = ctypes.CDLL(library_path)
    library.mingru_recurrent_scan.argtypes = [ctypes.c_void_p] * 14 + [ctypes.c_int] * 3
    library.mingru_recurrent_scan.restype = None

    torch.manual_seed(17)
    device = torch.device("cuda")
    batch_size, horizon, hidden_size = 2, 5, 3
    combined = torch.randn(
        batch_size,
        horizon,
        3 * hidden_size,
        device=device,
        requires_grad=True,
    )
    state = (torch.rand(batch_size, hidden_size, device=device) + 0.5).requires_grad_()
    inputs = torch.randn(
        batch_size,
        horizon,
        hidden_size,
        device=device,
        requires_grad=True,
    )
    episode_starts = torch.tensor(
        [[False, False, True, False, False], [True, False, False, False, True]],
        device=device,
    )
    expected_out, expected_next_state = _reference(
        combined,
        state,
        inputs,
        episode_starts,
    )
    grad_out = torch.randn_like(expected_out)
    grad_next_state = torch.randn_like(expected_next_state)
    torch.autograd.backward(
        (expected_out, expected_next_state),
        (grad_out, grad_next_state),
    )

    actual_out = torch.empty_like(expected_out)
    actual_next_state = torch.empty_like(expected_next_state)
    checkpoints = tuple(
        torch.empty(batch_size, horizon + 1, hidden_size, device=device)
        for _ in range(3)
    )
    actual_grad_combined = torch.empty_like(combined)
    actual_grad_state = torch.empty_like(state)
    actual_grad_input = torch.empty_like(inputs)
    tensors = (
        combined.detach(),
        state.detach(),
        inputs.detach(),
        episode_starts.float(),
        actual_out,
        actual_next_state,
        *checkpoints,
        grad_out,
        grad_next_state,
        actual_grad_combined,
        actual_grad_state,
        actual_grad_input,
    )
    library.mingru_recurrent_scan(
        *(_pointer(tensor) for tensor in tensors),
        batch_size,
        horizon,
        hidden_size,
    )

    torch.testing.assert_close(actual_out, expected_out, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(
        actual_next_state,
        expected_next_state,
        rtol=2e-4,
        atol=2e-4,
    )
    torch.testing.assert_close(
        actual_grad_combined,
        combined.grad,
        rtol=5e-4,
        atol=5e-4,
    )
    torch.testing.assert_close(actual_grad_state, state.grad, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(actual_grad_input, inputs.grad, rtol=5e-4, atol=5e-4)
