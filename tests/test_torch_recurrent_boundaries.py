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
