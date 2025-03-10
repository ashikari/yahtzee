import torch
from dataclasses import dataclass

from torch.distributions import Bernoulli, Categorical

import torch.nn.functional as F


class State:
    def __init__(self, batch_size: int):
        # current dice values encoded as the values of each die (ordered)
        self.dice_state = torch.zeros((batch_size, 6), dtype=torch.float32)
        # current dice values encoded as histogram
        # shape: batch, 6 (num dice)
        # the value of each element in this tensor is the number of dice with the associated value
        self.dice_histogram = torch.zeros((batch_size, 6), dtype=torch.float32)
        # rolls remaining
        # shape: batch, 1
        # the value is the number of rolls remaining
        self.rolls_remaining = torch.full((batch_size, 1), 3, dtype=torch.float32)

        # round index
        self.round_index = torch.full((batch_size, 1), 0, dtype=torch.float32)
        ## the current score sheet status

        # upper section:
        # Aces
        # twos
        # threes
        # fours
        # fives
        # sixes
        # values of current dice in each category
        # shape: batch, 6
        # value of the dice in each category
        self.upper_section_current_dice_scores = torch.zeros(
            (batch_size, 6), dtype=torch.float32
        )
        # which category are used
        # shape: batch, 6
        # value is 1 if the category was picked, value is 0 otherwise
        self.upper_section_used = torch.zeros((batch_size, 6), dtype=torch.float32)
        # the values of the used categories
        # shape: batch, 6
        # value is the value of the selected scores in each category
        self.upper_section_scores = torch.zeros((batch_size, 6), dtype=torch.float32)
        # the bonuses
        # shape: batch, 1
        # value is the value of the score
        self.upper_bonus = torch.zeros((batch_size, 1), dtype=torch.float32)
        # the total top score
        # batch, 1
        # value is the value of the upper section
        self.upper_score = torch.zeros((batch_size, 1), dtype=torch.float32)
        # lower section:
        # 3 of a kind
        # 4 of a kind
        # full house
        # small straight
        # large straight
        # Yahtzee
        # chance
        # values of current dice in each category
        # shape: batch, 7
        self.lower_section_current_dice_scores = torch.zeros(
            (batch_size, 7), dtype=torch.float32
        )
        # which goals are used
        # shape: batch, 7
        self.lower_section_used = torch.zeros((batch_size, 7), dtype=torch.float32)
        # the values of the used scores
        # shape: batch, 7
        self.lower_section_scores = torch.zeros((batch_size, 7), dtype=torch.float32)
        # Yahtzee Bonus
        # shape: batch, 1
        # value is the value of the score
        self.lower_bonus = torch.zeros((batch_size, 1), dtype=torch.float32)
        # the total lower score
        # shape: batch, 1
        self.lower_score = torch.zeros((batch_size, 1), dtype=torch.float32)
        # the total score across top and bottom
        # shape: batch, 1
        self.total_score = torch.zeros((batch_size, 1), dtype=torch.float32)

    def get_action_mask(self) -> torch.Tensor:
        return torch.cat(
            [self.upper_section_used, self.lower_section_used], dim=-1
        ).bool()

    def update_state_with_action(self, category_action: torch.Tensor) -> None:
        selected_mask = F.one_hot(category_action, num_classes=13)
        upper_selected_mask = selected_mask[:, :6]
        lower_selected_mask = selected_mask[:, 6:]

        # update upper section

        # section used
        self.upper_section_used += upper_selected_mask

        # section scores
        self.upper_section_scores += (
            upper_selected_mask * self.upper_section_current_dice_scores
        )

        # upper bonus
        upper_section_scores = torch.sum(
            self.upper_section_scores, dim=-1, keepdim=True
        )
        self.upper_bonus = torch.sum(self.upper_section_scores, dim=-1, keepdim=True)

        # upper score
        self.upper_score = upper_section_scores + self.upper_bonus

        # update lower section

        # section used
        self.lower_section_used += lower_selected_mask

        # section scores
        self.lower_section_scores = (
            lower_selected_mask * self.lower_section_current_dice_scores
        )

        # yahtzee bonus
        self.lower_bonus = (
            50 * torch.max(self.lower_section_scores, keepdim=True, dim=-1).values > 1
        )

        # lower score
        self.lower_score = (
            self.lower_section_scores.sum(dim=-1, keepdim=True) + self.lower_bonus
        )

        # total score
        self.total_score = self.upper_score + self.lower_score

    def get_feature_vector(self) -> torch.Tensor:
        return torch.cat(
            [
                self.dice_state,
                self.dice_histogram,
                self.rolls_remaining,
                self.round_index,
                self.upper_section_current_dice_scores,
                self.upper_section_used,
                self.upper_section_scores,
                self.upper_bonus,
                self.upper_score,
                self.lower_section_current_dice_scores,
                self.lower_section_used,
                self.lower_section_scores,
                self.lower_score,
                self.total_score,
            ],
            dim=1,
        )


@dataclass
class Action:
    # indicates which dice to re-roll
    dice_action: torch.Tensor
    # indicates which category to pick for that round
    category_action: torch.Tensor

    def sample_dice_action(self):
        return Bernoulli(self.dice_action).sample()

    def sample_category_action(self, state: State):
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
