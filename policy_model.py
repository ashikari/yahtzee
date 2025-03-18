import torch
from dataclasses import dataclass

from torch.distributions import Bernoulli, Categorical

import torch.nn.functional as F
from typing import List

from state import State


@dataclass
class Action:
    # indicates which dice to re-roll
    dice_action: torch.Tensor
    # Logits indicating which category to pick for that round
    category_action: torch.Tensor

    def sample_dice_action(self):
        return Bernoulli(logits=self.dice_action).sample()

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

    def clone(self) -> "Action":
        return Action(
            dice_action=self.dice_action.clone(),
            category_action=self.category_action.clone(),
        )


class MLP(torch.nn.Module):
    def __init__(self, layers: List[int], final_activation: bool = False):
        super().__init__()
        modules = []
        for l_idx in range(1, len(layers)):
            modules.append(
                torch.nn.Linear(
                    in_features=layers[l_idx - 1], out_features=layers[l_idx]
                ),
            )
            if l_idx < len(layers) - 1 or (
                l_idx == len(layers) - 1 and final_activation
            ):
                modules.append(torch.nn.ReLU(inplace=True))

        self.mlp = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        return out


class PolicyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_mlp = MLP(
            [State.get_feature_length(), 128, 128], final_activation=False
        )
        self.dice_mlp = MLP([128, 64, 5], final_activation=False)
        self.category_mlp = MLP([128, 64, 13], final_activation=False)

    def forward(self, state_vector: torch.Tensor) -> Action:
        shared_embedding = self.shared_mlp(state_vector)
        dice_action = self.dice_mlp(shared_embedding)
        category_action = self.category_mlp(shared_embedding)

        # dummy action where the dice are not rolled and the category is random
        return Action(
            dice_action=dice_action,
            category_action=category_action,
        )
