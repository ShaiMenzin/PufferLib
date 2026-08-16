import torch
from pufferlib.models import MinGRU, reset_recurrent_state


def test_mingru_training_respects_initial_state_and_terminal_boundaries() -> None:
    torch.manual_seed(7)
    network = MinGRU(hidden_size=4, num_layers=2)
    inputs = torch.randn(3, 5, 4)
    initial_state = (torch.randn(2, 3, 4),)
    terminals = torch.zeros(3, 5, dtype=torch.bool)
    terminals[0, 2] = True
    terminals[2, 4] = True

    expected_steps = []
    state = tuple(value.clone() for value in initial_state)
    for step in range(inputs.shape[1]):
        state = reset_recurrent_state(state, terminals[:, step])
        output, state = network.forward_eval(inputs[:, step], state)
        expected_steps.append(output)

    actual = network.forward_train_recurrent(inputs, initial_state, terminals)

    torch.testing.assert_close(actual, torch.stack(expected_steps, dim=1))


def test_mingru_training_without_boundaries_matches_recurrent_steps() -> None:
    torch.manual_seed(11)
    network = MinGRU(hidden_size=4, num_layers=2)
    inputs = torch.randn(3, 5, 4)
    initial_state = (torch.rand(2, 3, 4),)
    terminals = torch.zeros(3, 5, dtype=torch.bool)

    expected_steps = []
    state = tuple(value.clone() for value in initial_state)
    for step in range(inputs.shape[1]):
        output, state = network.forward_eval(inputs[:, step], state)
        expected_steps.append(output)

    actual = network.forward_train_recurrent(inputs, initial_state, terminals)

    torch.testing.assert_close(actual, torch.stack(expected_steps, dim=1))


def test_mingru_training_without_boundaries_preserves_negative_initial_state() -> None:
    torch.manual_seed(13)
    network = MinGRU(hidden_size=4, num_layers=2)
    inputs = torch.randn(2, 4, 4)
    initial_state = (torch.randn(2, 2, 4),)
    terminals = torch.zeros(2, 4, dtype=torch.bool)
    assert bool((initial_state[0] < 0).any())

    expected_steps = []
    state = tuple(value.clone() for value in initial_state)
    for step in range(inputs.shape[1]):
        output, state = network.forward_eval(inputs[:, step], state)
        expected_steps.append(output)

    actual = network.forward_train_recurrent(inputs, initial_state, terminals)

    torch.testing.assert_close(actual, torch.stack(expected_steps, dim=1))
