import torch
import unittest
from state import State
from score import (
    three_of_a_kind,
    four_of_a_kind,
    full_house,
    small_straight,
    large_straight,
    yahtzee,
    chance,
    compute_scores,
    _detect_straight,
)


class TestScore(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_test_state(self, batch_size, dice_values):
        """Helper to create a test state with specific dice values"""
        state = State(batch_size, self.device)

        # Set dice values for each batch element
        for i, dice in enumerate(dice_values):
            if i >= batch_size:
                break
            state.dice_state[i] = torch.tensor(dice, device=self.device)

            # Update histogram
            histogram = torch.zeros(6, device=self.device)
            for die in dice:
                if die > 0:  # Skip zeros (which represent unset dice)
                    histogram[int(die) - 1] += 1
            state.dice_histogram[i] = histogram

        return state

    def test_three_of_a_kind(self):
        # Create test cases: [has 3oak, doesn't have 3oak, has 4oak (should count as 3oak too)]
        dice_values = [
            [3, 3, 3, 1, 2],  # Has 3 of a kind (sum = 12)
            [1, 2, 3, 4, 5],  # No 3 of a kind (sum = 0)
            [4, 4, 4, 4, 2],  # Has 4 of a kind (should count as 3oak too, sum = 18)
        ]

        state = self.create_test_state(3, dice_values)
        result = three_of_a_kind(state)

        self.assertEqual(result.shape, (3, 1))
        self.assertEqual(
            result[0, 0].item(), 12
        )  # Sum of all dice for the first test case
        self.assertEqual(result[1, 0].item(), 0)  # No 3 of a kind
        self.assertEqual(
            result[2, 0].item(), 18
        )  # Sum of all dice for the third test case

    def test_four_of_a_kind(self):
        # Create test cases: [has 4oak, has 3oak but not 4oak, has 5oak (yahtzee)]
        dice_values = [
            [2, 2, 2, 2, 5],  # Has 4 of a kind (sum = 13)
            [3, 3, 3, 1, 2],  # Has 3 of a kind but not 4 (sum = 0)
            [6, 6, 6, 6, 6],  # Has 5 of a kind (sum = 30)
        ]

        state = self.create_test_state(3, dice_values)
        result = four_of_a_kind(state)

        self.assertEqual(result.shape, (3, 1))
        self.assertEqual(result[0, 0].item(), 13)  # Sum of all dice
        self.assertEqual(result[1, 0].item(), 0)  # No 4 of a kind
        self.assertEqual(result[2, 0].item(), 30)  # Sum of all dice

    def test_full_house(self):
        # Create test cases: [has full house, has 3oak but not full house, has 4oak but not full house, has yahtzee]
        dice_values = [
            [2, 2, 2, 5, 5],  # Has full house (3 of 2s, 2 of 5s)
            [3, 3, 3, 1, 2],  # Has 3 of a kind but not full house
            [4, 4, 4, 4, 2],  # Has 4 of a kind but not full house
            [6, 6, 6, 6, 6],  # Has yahtzee (not a traditional full house)
        ]

        state = self.create_test_state(4, dice_values)
        result = full_house(state)

        self.assertEqual(result.shape, (4, 1))
        self.assertEqual(result[0, 0].item(), 25)  # Full house score
        self.assertEqual(result[1, 0].item(), 0)  # Not a full house
        self.assertEqual(result[2, 0].item(), 0)  # Not a full house
        self.assertEqual(
            result[3, 0].item(), 0
        )  # Yahtzee is not a full house by standard rules

    def test_detect_straight(self):
        # Test cases for the helper function
        dice_values = [
            [1, 2, 3, 4, 6],  # Small straight (4 consecutive)
            [2, 3, 4, 5, 6],  # Large straight (5 consecutive)
            [1, 3, 4, 5, 6],  # Has small straight (3-4-5-6) despite missing 2
            [1, 2, 5, 6, 6],  # No small straight (gaps prevent 4 consecutive)
        ]

        state = self.create_test_state(4, dice_values)

        # Test small straight detection (length 4)
        small_result = _detect_straight(state, 4)
        self.assertEqual(small_result.shape, (4, 1))
        self.assertEqual(small_result[0, 0].item(), 1)  # Has small straight
        self.assertEqual(
            small_result[1, 0].item(), 1
        )  # Large straight includes small straight
        self.assertEqual(
            small_result[2, 0].item(), 1
        )  # Should detect 3-4-5-6 as small straight
        self.assertEqual(small_result[3, 0].item(), 0)  # No small straight

        # Test large straight detection (length 5)
        large_result = _detect_straight(state, 5)
        self.assertEqual(large_result.shape, (4, 1))
        self.assertEqual(large_result[0, 0].item(), 0)  # Not a large straight
        self.assertEqual(large_result[1, 0].item(), 1)  # Has large straight
        self.assertEqual(large_result[2, 0].item(), 0)  # Gap prevents large straight
        self.assertEqual(large_result[3, 0].item(), 0)  # Not a large straight

    def test_small_straight(self):
        # Create test cases: [has small straight, has large straight, no straight]
        dice_values = [
            [1, 2, 3, 4, 1],  # Has small straight
            [1, 2, 3, 4, 5],  # Has large straight (includes small)
            [1, 2, 4, 5, 6],  # No small straight (gap at 3)
        ]

        state = self.create_test_state(3, dice_values)
        result = small_straight(state)

        self.assertEqual(result.shape, (3, 1))
        self.assertEqual(result[0, 0].item(), 30)  # Small straight score
        self.assertEqual(
            result[1, 0].item(), 30
        )  # Large straight includes small straight
        self.assertEqual(result[2, 0].item(), 0)  # No small straight

    def test_large_straight(self):
        # Create test cases: [has large straight, has small straight but not large, no straight]
        dice_values = [
            [1, 2, 3, 4, 5],  # Has large straight
            [2, 3, 4, 5, 5],  # Has small straight but not large
            [1, 3, 4, 5, 6],  # No large straight (gap at 2)
        ]

        state = self.create_test_state(3, dice_values)
        result = large_straight(state)

        self.assertEqual(result.shape, (3, 1))
        self.assertEqual(result[0, 0].item(), 40)  # Large straight score
        self.assertEqual(result[1, 0].item(), 0)  # No large straight
        self.assertEqual(result[2, 0].item(), 0)  # No large straight

    def test_yahtzee(self):
        # Create test cases: [has yahtzee, has 4oak but not yahtzee, no yahtzee]
        dice_values = [
            [4, 4, 4, 4, 4],  # Has yahtzee
            [3, 3, 3, 3, 1],  # Has 4 of a kind but not yahtzee
            [1, 2, 3, 4, 5],  # No yahtzee
        ]

        state = self.create_test_state(3, dice_values)
        result = yahtzee(state)

        self.assertEqual(result.shape, (3, 1))
        self.assertEqual(result[0, 0].item(), 50)  # Yahtzee score
        self.assertEqual(result[1, 0].item(), 0)  # No yahtzee
        self.assertEqual(result[2, 0].item(), 0)  # No yahtzee

    def test_chance(self):
        # Create test cases with different dice sums
        dice_values = [
            [1, 1, 1, 1, 1],  # Sum = 5
            [6, 6, 6, 6, 6],  # Sum = 30
            [1, 2, 3, 4, 5],  # Sum = 15
        ]

        state = self.create_test_state(3, dice_values)
        result = chance(state)

        self.assertEqual(result.shape, (3, 1))
        self.assertEqual(result[0, 0].item(), 5)  # Sum of dice
        self.assertEqual(result[1, 0].item(), 30)  # Sum of dice
        self.assertEqual(result[2, 0].item(), 15)  # Sum of dice

    def test_compute_scores(self):
        # Create a test case with various dice configurations
        dice_values = [
            [1, 1, 1, 2, 3],  # 3 of a kind of 1s
            [2, 3, 4, 5, 6],  # Large straight
            [6, 6, 6, 6, 6],  # Yahtzee of 6s
        ]

        state = self.create_test_state(3, dice_values)
        result_state = compute_scores(state)

        # Check that the state was updated correctly

        # Upper section scores
        self.assertEqual(
            result_state.upper_section_current_dice_scores[0, 0].item(), 3
        )  # Three 1s
        self.assertEqual(
            result_state.upper_section_current_dice_scores[1, 5].item(), 6
        )  # One 6
        self.assertEqual(
            result_state.upper_section_current_dice_scores[2, 5].item(), 30
        )  # Five 6s

        # Lower section scores
        # 3oak, 4oak, full house, small straight, large straight, yahtzee, chance

        # First row (3 of a kind of 1s)
        self.assertEqual(
            result_state.lower_section_current_dice_scores[0, 0].item(), 8
        )  # 3oak sum
        self.assertEqual(
            result_state.lower_section_current_dice_scores[0, 1].item(), 0
        )  # No 4oak
        self.assertEqual(
            result_state.lower_section_current_dice_scores[0, 2].item(), 0
        )  # No full house
        self.assertEqual(
            result_state.lower_section_current_dice_scores[0, 3].item(), 0
        )  # No small straight
        self.assertEqual(
            result_state.lower_section_current_dice_scores[0, 4].item(), 0
        )  # No large straight
        self.assertEqual(
            result_state.lower_section_current_dice_scores[0, 5].item(), 0
        )  # No yahtzee
        self.assertEqual(
            result_state.lower_section_current_dice_scores[0, 6].item(), 8
        )  # Chance sum

        # Second row (large straight)
        self.assertEqual(
            result_state.lower_section_current_dice_scores[1, 0].item(), 0
        )  # No 3oak
        self.assertEqual(
            result_state.lower_section_current_dice_scores[1, 1].item(), 0
        )  # No 4oak
        self.assertEqual(
            result_state.lower_section_current_dice_scores[1, 2].item(), 0
        )  # No full house
        self.assertEqual(
            result_state.lower_section_current_dice_scores[1, 3].item(), 30
        )  # Small straight
        self.assertEqual(
            result_state.lower_section_current_dice_scores[1, 4].item(), 40
        )  # Large straight
        self.assertEqual(
            result_state.lower_section_current_dice_scores[1, 5].item(), 0
        )  # No yahtzee
        self.assertEqual(
            result_state.lower_section_current_dice_scores[1, 6].item(), 20
        )  # Chance sum

        # Third row (yahtzee of 6s)
        self.assertEqual(
            result_state.lower_section_current_dice_scores[2, 0].item(), 30
        )  # 3oak sum
        self.assertEqual(
            result_state.lower_section_current_dice_scores[2, 1].item(), 30
        )  # 4oak sum
        self.assertEqual(
            result_state.lower_section_current_dice_scores[2, 2].item(), 0
        )  # No full house (all same)
        self.assertEqual(
            result_state.lower_section_current_dice_scores[2, 3].item(), 0
        )  # No small straight
        self.assertEqual(
            result_state.lower_section_current_dice_scores[2, 4].item(), 0
        )  # No large straight
        self.assertEqual(
            result_state.lower_section_current_dice_scores[2, 5].item(), 50
        )  # Yahtzee
        self.assertEqual(
            result_state.lower_section_current_dice_scores[2, 6].item(), 30
        )  # Chance sum


if __name__ == "__main__":
    unittest.main()
