from policy_model import State
import torch


def three_of_a_kind(state: State) -> torch.Tensor:
    max_histogram = torch.max(state.dice_histogram, dim=1, keepdim=True).values
    return (max_histogram >= 3) * state.dice_state.sum(dim=1, keepdim=True)


def four_of_a_kind(state: State) -> torch.Tensor:
    max_histogram = torch.max(state.dice_histogram, dim=1, keepdim=True).values
    return (max_histogram >= 4) * state.dice_state.sum(dim=1, keepdim=True)


def full_house(state: State) -> torch.Tensor:
    sorted_histogram = torch.sort(state.dice_histogram, dim=1, descending=True).values
    has_three = sorted_histogram[:, 0:1] >= 3
    has_two = sorted_histogram[:, 1:2] >= 2
    return (has_three & has_two) * 25


def _detect_straight(state: State, length: int) -> torch.Tensor:
    """
    Helper function to detect straights of a given length.

    Args:
        state: The current game state
        length: The length of the straight to detect (4 for small, 5 for large)

    Returns:
        A tensor indicating whether each batch element has the straight
    """
    # Convert histogram to binary presence (1 if at least one die shows the value)
    dice_present = (state.dice_histogram > 0).float()

    # Use convolution to detect sequences of consecutive values
    consecutive_count = torch.conv1d(
        dice_present.unsqueeze(1),  # Add channel dimension
        torch.ones(
            1, 1, length, device=dice_present.device
        ),  # Kernel to detect consecutive 1s
        padding=0,
    ).squeeze(1)

    # If any position has a value equal to the length, we have a straight
    return (consecutive_count >= length).any(dim=1, keepdim=True)


def small_straight(state: State) -> torch.Tensor:
    # Check for small straight (sequence of 4)
    is_small_straight = _detect_straight(state, 4)
    return is_small_straight * 30


def large_straight(state: State) -> torch.Tensor:
    # Check for large straight (sequence of 5)
    is_large_straight = _detect_straight(state, 5)
    return is_large_straight * 40


def yahtzee(state: State) -> torch.Tensor:
    # Check for Yahtzee (all 5 dice showing the same value)
    return (state.dice_histogram.max(dim=1, keepdim=True).values == 5) * 50


def chance(state: State) -> torch.Tensor:
    # Sum of all dice
    return state.dice_state.sum(dim=1, keepdim=True)


def compute_scores(state: State) -> State:
    state.upper_section_current_dice_scores = state.dice_histogram * torch.arange(
        1, 7
    ).reshape(1, 6)

    state.lower_section_current_dice_scores = torch.cat(
        [
            three_of_a_kind(state),
            four_of_a_kind(state),
            full_house(state),
            small_straight(state),
            large_straight(state),
            yahtzee(state),
            chance(state),
        ],
        dim=1,
    )
    return state
