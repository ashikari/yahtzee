from state import State
from policy_model import Action, PolicyModel
from typing import Optional, List, Tuple
import torch
from score import compute_scores


class Yahtzee(torch.nn.Module):
    def __init__(self, batch_size: int, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        # Game params that do not change
        self.num_dice = 5
        self.num_rounds = 13
        self.dice_sides = 6

        # initialize policy model
        self.policy_model = PolicyModel().to(self.device)

    def forward(self) -> Tuple[List[State], List[Action]]:
        """
        Plays the entire game of Yahtzee.

        Returns:
            Tuple[List[State], List[Action]]: A tuple of a list of states representing each step in the round and a list of actions taken
        """
        states = []
        actions = []
        values = []

        state = State(self.batch_size, device=self.device)

        for round_idx in range(self.num_rounds):
            # set round index
            state.round_index = torch.full(
                (self.batch_size, 1), round_idx, device=state.dice_state.device
            )
            round_states, round_actions, round_values = self.play_round(state)
            states.extend(round_states)
            actions.extend(round_actions)
            values.extend(round_values)

        # compute rewards
        rewards = self.compute_rewards(states, actions)
        values = torch.cat(values, dim=1)

        return states, actions, values, rewards

    def compute_rewards(
        self, states: List[State], actions: List[Action]
    ) -> torch.Tensor:
        """
        Computes the rewards for the game based on score changes and final score.

        Args:
            states: List[State] - A list of game states at each step
            actions: List[Action] - A list of actions taken at each step

        Returns:
            torch.Tensor: Rewards tensor with shape (batch_size, num_steps) where:
                - Each row represents a game instance in the batch
                - Each column represents the reward at a particular step
                - Rewards are based on score increases between states and a large
                  bonus for the final score
        """

        total_scores = [state.total_score for state in states]
        total_scores = torch.cat(total_scores, dim=1)
        # Calculate score deltas (negative because we want to reward increases)
        score_delta = torch.zeros_like(total_scores)
        score_delta[:, 1:] = total_scores[:, 1:] - total_scores[:, :-1]

        final_score_reward = torch.zeros_like(total_scores)
        final_score_reward[:, -1] = total_scores[:, -1]

        return score_delta + final_score_reward * 5

    def play_round(self, state: State) -> Tuple[List[State], List[Action]]:
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
            Tuple[List[State], List[Action]]: A tuple of a list of states representing each step in the round and a list of actions taken
        """
        states = []
        actions = []
        values = []

        # first roll
        state = self.roll_dice(state)
        # TODO: remove this logging outside of debug mode
        states.append(state.clone())
        actions.append(Action(None, None, None))

        # second and third rolls
        for _ in range(1, 3):
            a, value = self.policy_model(state.get_feature_vector())
            state = self.roll_dice(state, a.sample_dice_action())

            actions.append(a.clone())
            states.append(state.clone())
            values.append(value)

        a, value = self.policy_model(state.get_feature_vector())
        state = self.select_categories(state, a)
        actions.append(a.clone())
        states.append(state.clone())
        values.append(value)
        return states, actions, values

    def select_categories(self, state: State, action: Action) -> State:
        """
        Selects the categories for the current round.

        Args:
            state: The current game state
            action: The action to select categories

        Returns:
            Updated state with selected categories
        """
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
        # Generate all dice rolls at once for better GPU parallelization
        # Using float32 directly avoids type conversion later
        new_rolls = torch.randint(
            1,
            self.dice_sides + 1,
            (self.batch_size, self.num_dice),
            dtype=torch.float32,  # Match target dtype to avoid conversion
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
            index=state.dice_state.long() - 1,
            src=torch.ones(
                state.dice_state.shape,
                dtype=torch.float32,
                device=state.dice_state.device,
            ),
        )

        # score the state
        state = compute_scores(state)
        return state
