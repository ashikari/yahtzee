from state import State
from policy_model import Action, PolicyModel
from typing import Optional, List
import torch
from score import compute_scores


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
    def forward(self) -> List[State]:
        states = []
        state = State(self.batch_size)

        for round_idx in range(self.num_rounds):
            # set round index
            state.round_index = torch.full(
                (self.batch_size, 1), round_idx, device=state.dice_state.device
            )
            round_states = self.play_round(state)
            states.extend(round_states)

        return states

    def play_round(self, state: State) -> List[State]:
        """
        Plays a single round of Yahtzee, consisting of up to three dice rolls followed by category selection.

        A round consists of:
        1. First roll (all dice are rolled)
        2. Second roll (optional, player chooses which dice to re-roll)
        3. Third roll (optional, player chooses which dice to re-roll)
        4. Category selection (player selects which scoring category to use)

        Args:
            state: The current game state

        Returns:
            List[State]: A list of states representing each step in the round
        """
        states = []

        # first roll
        state = self.roll_dice(state)
        states.append(state.clone())

        # second and third rolls
        for _ in range(1, 3):
            a = self.policy_model(state.get_feature_vector())
            state = self.roll_dice(state, a.sample_dice_action())
            states.append(state.clone())

        a = self.policy_model(state.get_feature_vector())
        state = self.select_categories(state, a)
        states.append(state.clone())
        return states

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
            state.rolls_remaining -= 1
        else:
            # first roll in a round
            state.dice_state = new_rolls
            state.rolls_remaining = torch.full(
                (self.batch_size, 1), 2, device=state.device
            )
        state.dice_state, _ = torch.sort(state.dice_state, dim=1)

        # update dice histogram
        state.dice_histogram = torch.zeros(
            (self.batch_size, self.dice_sides), device=state.dice_state.device
        )
        state.dice_histogram.scatter_add_(
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
