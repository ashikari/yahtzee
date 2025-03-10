from policy_model import State, Action
from typing import Optional
import torch
from score import compute_scores

from policy_model import PolicyModel


class Yahtzee(torch.nn.Module):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

        # Game params that do not change
        self.num_dice = 5
        self.num_rounds = 13
        self.dice_sides = 6

        # initialize policy model
        self.policy_model = PolicyModel()

    # TODO: update outputs to return all necessary inputs to compute rewards
    def forward(self) -> None:
        state = State(self.batch_size)

        for round_idx in range(self.num_rounds):
            # set round index
            state.round_index = torch.full(
                (self.batch_size, 1), round_idx, device=state.dice_state.device
            )
            self.play_round(state)

    def play_round(self, state: State) -> None:
        state = self.roll_dice(state)

        for roll_idx in range(2):
            state.rolls_remaining = torch.full(
                (self.batch_size, 1), 2 - roll_idx, device=state.dice_state.device
            )
            a = self.policy_model(state.get_feature_vector())
            state = self.roll_dice(state, a.sample_dice_action())

        a = self.policy_model(state.get_feature_vector())
        state = self.select_categories(state, a)

    def select_categories(self, state: State, action: Action) -> State:
        # get action from category sample
        category_action = action.sample_category_action(state)
        state.update_state_with_action(category_action)
        return state

    def roll_dice(self, state: State, dice_action: Optional[Action] = None) -> State:
        """
        Roll the dice according to the dice action mask.

        Args:
            state: The current game state
            dice_action: Optional mask tensor indicating which dice to re-roll.
                        If None, rolls all dice. If provided, True values indicate
                        dice positions to re-roll, False values keep current dice.

        Returns:
            Updated state with new dice values
        """
        # simulate rolling all 5 dice.
        new_rolls = torch.randint(
            1,
            self.dice_sides + 1,
            (self.batch_size, self.num_dice),
            device=state.dice_state.device,
        )

        # update dice state
        if dice_action is not None:
            state.dice_state = torch.where(
                dice_action.bool(), new_rolls, state.dice_state
            )
        else:
            # first roll in a round
            state.dice_state = new_rolls
        torch.sort(state.dice_state, dim=1)

        # update dice histogram
        state.dice_histagram = torch.zeros(
            (self.batch_size, self.dice_sides), device=state.dice_state.device
        )
        state.dice_histagram.scatter_add(
            dim=1,
            index=state.dice_state - 1,
            src=torch.ones(
                state.dice_state.shape,
                dtype=torch.float32,
                device=state.dice_state.device,
            ),
        )

        # score the state
        state = compute_scores(state)
        return state
