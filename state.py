import torch
import torch.nn.functional as F


class State:
    def __init__(self, batch_size: int, device: torch.device = torch.device("cpu")):
        self.batch_size = batch_size
        self.device = device
        # current dice values encoded as the values of each die (ordered)
        self.dice_state = torch.zeros(
            (batch_size, 5), dtype=torch.float32, device=device
        )
        # current dice values encoded as histogram
        # shape: batch, 6 (num dice)
        # the value of each element in this tensor is the number of dice with the associated value
        self.dice_histogram = torch.zeros(
            (batch_size, 6), dtype=torch.float32, device=device
        )
        # rolls remaining
        # shape: batch, 1
        # the value is the number of rolls remaining
        self.rolls_remaining = torch.full(
            (batch_size, 1), 3, dtype=torch.float32, device=device
        )

        # round index
        self.round_index = torch.full(
            (batch_size, 1), 0, dtype=torch.float32, device=device
        )
        ## the current score sheet status

        # upper section scores (Aces through Sixes)
        # shape: batch, 6
        self.upper_section_current_dice_scores = torch.zeros(
            (batch_size, 6), dtype=torch.float32, device=device
        )
        # which category are used
        # shape: batch, 6
        # value is 1 if the category was picked, value is 0 otherwise
        self.upper_section_used = torch.zeros(
            (batch_size, 6), dtype=torch.float32, device=device
        )
        # the values of the used categories
        # shape: batch, 6
        # value is the value of the selected scores in each category
        self.upper_section_scores = torch.zeros(
            (batch_size, 6), dtype=torch.float32, device=device
        )
        # the bonuses
        # shape: batch, 1
        # value is the value of the score
        self.upper_bonus = torch.zeros(
            (batch_size, 1), dtype=torch.float32, device=device
        )
        # the total top score
        # batch, 1
        # value is the value of the upper section
        self.upper_score = torch.zeros(
            (batch_size, 1), dtype=torch.float32, device=device
        )
        # lower section scores for current dice (3 of a kind, 4 of a kind, full house,
        # small straight, large straight, Yahtzee, chance)
        # shape: batch, 7
        self.lower_section_current_dice_scores = torch.zeros(
            (batch_size, 7), dtype=torch.float32, device=device
        )
        # which goals are used
        # shape: batch, 7
        self.lower_section_used = torch.zeros(
            (batch_size, 7), dtype=torch.float32, device=device
        )
        # the values of the used scores
        # shape: batch, 7
        self.lower_section_scores = torch.zeros(
            (batch_size, 7), dtype=torch.float32, device=device
        )
        # Yahtzee Bonus
        # shape: batch, 1
        # value is the value of the score
        self.lower_bonus = torch.zeros(
            (batch_size, 1), dtype=torch.float32, device=device
        )
        # the total lower score
        # shape: batch, 1
        self.lower_score = torch.zeros(
            (batch_size, 1), dtype=torch.float32, device=device
        )
        # the total score across top and bottom
        # shape: batch, 1
        self.total_score = torch.zeros(
            (batch_size, 1), dtype=torch.float32, device=device
        )

    def get_action_mask(self) -> torch.Tensor:
        """
        Returns a boolean mask indicating which categories have already been used.

        The mask combines both upper and lower section usage information.
        True values indicate categories that have been used, while False values
        indicate categories that are still available for selection.

        Returns:
            torch.Tensor: A boolean tensor with shape (batch_size, 13) where
                         indices 0-5 represent upper section categories and
                         indices 6-12 represent lower section categories.
        """
        return torch.cat(
            [self.upper_section_used, self.lower_section_used], dim=-1
        ).bool()

    def update_state_with_action(self, category_action: torch.Tensor) -> None:
        """
        Updates the game state based on the selected category action.

        This method processes the category selection and updates all relevant
        game state components including:
        - Upper section scores and usage
        - Lower section scores and usage
        - Bonus calculations
        - Total score

        Args:
            category_action: Tensor containing the indices of selected categories
                            (0-5 for upper section, 6-12 for lower section)
        """
        # Convert category indices to one-hot encoding
        selected_mask = F.one_hot(category_action, num_classes=13)
        upper_selected_mask = selected_mask[:, :6]  # Aces through Sixes
        lower_selected_mask = selected_mask[:, 6:]  # 3-of-a-kind through Chance

        # Update upper section
        self.upper_section_used += upper_selected_mask
        self.upper_section_scores += (
            upper_selected_mask * self.upper_section_current_dice_scores
        )

        # Calculate upper section bonus (35 points if sum ≥ 63)
        upper_section_sum = torch.sum(self.upper_section_scores, dim=-1, keepdim=True)
        self.upper_bonus = (upper_section_sum >= 63).float() * 35

        # Calculate total upper section score
        self.upper_score = upper_section_sum + self.upper_bonus

        # Update lower section
        self.lower_section_used += lower_selected_mask
        self.lower_section_scores += (
            lower_selected_mask * self.lower_section_current_dice_scores
        )

        # Add 100 points for each additional Yahtzee
        self.lower_bonus += self._yahtzee_bonus_condition(lower_selected_mask) * 100

        # Calculate total lower section score
        self.lower_score = (
            self.lower_section_scores.sum(dim=-1, keepdim=True) + self.lower_bonus
        )

        # Calculate grand total
        self.total_score = self.upper_score + self.lower_score

    def _yahtzee_bonus_condition(
        self, lower_selected_mask: torch.Tensor
    ) -> torch.Tensor:
        # Calculate Yahtzee bonus (100 points for each additional Yahtzee beyond the first)
        yahtzee_idx = 5  # Index of Yahtzee in lower section (0-indexed)

        # Check if current selection is Yahtzee category
        is_yahtzee_selected = lower_selected_mask[:, yahtzee_idx : yahtzee_idx + 1] == 1

        # Check if current dice form a Yahtzee
        has_current_yahtzee = (
            self.lower_section_current_dice_scores[:, yahtzee_idx : yahtzee_idx + 1] > 0
        )

        # Check if a Yahtzee was previously scored (with 50 points)
        has_previous_yahtzee = (
            self.lower_section_scores[:, yahtzee_idx : yahtzee_idx + 1] == 50
        )

        # Award bonus only if:
        # 1. Yahtzee category is selected this turn
        # 2. Current dice form a Yahtzee
        # 3. A Yahtzee was previously scored with 50 points
        yahtzee_bonus_condition = (
            is_yahtzee_selected & has_current_yahtzee & has_previous_yahtzee
        )
        return yahtzee_bonus_condition.float()

    def get_feature_vector(self) -> torch.Tensor:
        """
        Constructs a feature vector representing the current game state.

        This method concatenates all relevant state information into a single tensor
        that can be used as input to the policy model.

        Returns:
            torch.Tensor: A concatenated tensor containing all state features
        """
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

    def clone(self) -> "State":
        cloned_state = State(batch_size=self.batch_size, device=self.device)
        cloned_state.dice_state = torch.clone(self.dice_state)
        cloned_state.dice_histogram = torch.clone(self.dice_histogram)
        cloned_state.rolls_remaining = torch.clone(self.rolls_remaining)
        cloned_state.round_index = torch.clone(self.round_index)
        cloned_state.upper_section_current_dice_scores = torch.clone(
            self.upper_section_current_dice_scores
        )
        cloned_state.upper_section_used = torch.clone(self.upper_section_used)
        cloned_state.upper_section_scores = torch.clone(self.upper_section_scores)
        cloned_state.upper_bonus = torch.clone(self.upper_bonus)
        cloned_state.upper_score = torch.clone(self.upper_score)
        cloned_state.lower_section_current_dice_scores = torch.clone(
            self.lower_section_current_dice_scores
        )
        cloned_state.lower_section_used = torch.clone(self.lower_section_used)
        cloned_state.lower_section_scores = torch.clone(self.lower_section_scores)
        cloned_state.lower_bonus = torch.clone(self.lower_bonus)
        cloned_state.lower_score = torch.clone(self.lower_score)
        cloned_state.total_score = torch.clone(self.total_score)

        return cloned_state

    def to(self, device: torch.device) -> "State":
        """
        Moves all tensors in the state to the specified device.

        Args:
            device: The target device to move tensors to

        Returns:
            self: The state object with all tensors moved to the specified device
        """
        self.device = device
        self.dice_state = self.dice_state.to(device)
        self.dice_histogram = self.dice_histogram.to(device)
        self.rolls_remaining = self.rolls_remaining.to(device)
        self.round_index = self.round_index.to(device)
        self.upper_section_current_dice_scores = (
            self.upper_section_current_dice_scores.to(device)
        )
        self.upper_section_used = self.upper_section_used.to(device)
        self.upper_section_scores = self.upper_section_scores.to(device)
        self.upper_bonus = self.upper_bonus.to(device)
        self.upper_score = self.upper_score.to(device)
        self.lower_section_current_dice_scores = (
            self.lower_section_current_dice_scores.to(device)
        )
        self.lower_section_used = self.lower_section_used.to(device)
        self.lower_section_scores = self.lower_section_scores.to(device)
        self.lower_bonus = self.lower_bonus.to(device)
        self.lower_score = self.lower_score.to(device)
        self.total_score = self.total_score.to(device)

        return self

    @staticmethod
    def get_feature_length() -> int:
        """
        Calculates the total length of the feature vector.

        This method computes the sum of the second dimension sizes of all tensors
        that are concatenated in the get_feature_vector method.

        Returns:
            int: The total length of the feature vector
        """
        # Dice state: 6
        # Dice histogram: 6
        # Rolls remaining: 1
        # Round index: 1
        # Upper section current dice scores: 6
        # Upper section used: 6
        # Upper section scores: 6
        # Upper bonus: 1
        # Upper score: 1
        # Lower section current dice scores: 7
        # Lower section used: 7
        # Lower section scores: 7
        # Lower score: 1
        # Total score: 1
        return 6 + 6 + 1 + 1 + 6 + 6 + 6 + 1 + 1 + 7 + 7 + 7 + 1  # Total: 57
