import torch
from dataclasses import dataclass

from torch.distributions import Bernoulli, Categorical

import torch.nn.functional as F

from state import State


@dataclass
class Action:
    # indicates which dice to re-roll
    dice_action: torch.Tensor
    # Logits indicating which category to pick for that round
    category_action: torch.Tensor

    def sample_dice_action(self):
        return Bernoulli(self.dice_action).sample()

    def sample_category_action(self, state: State):
        """
        Samples a category action based on the current state.

        This method applies a mask to the category action logits to prevent selecting
        categories that have already been used. It then samples from the resulting
        categorical distribution to determine which category to select.

        Args:
            state: The current game state containing information about used categories.

        Returns:
            A tensor representing the sampled category action.
        """
        mask = state.get_action_mask()
        self.category_action = self.category_action.masked_fill(mask, -1e9)
        category_action_sample = Categorical(logits=self.category_action).sample()
        return category_action_sample


class PolicyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state_vector: torch.Tensor) -> Action:
        batch_size = state_vector.shape[0]

        # dummy action where the dice are not rolled and the category is random
        return Action(
            dice_action=torch.zeros((batch_size, 5), dtype=torch.float32),
            category_action=torch.rand((batch_size, 13), dtype=torch.float32),
        )
