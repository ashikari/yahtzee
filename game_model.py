from policy_model import State
import torch

from score import score
from policy_model import PolicyModel

class Yahtzee(torch.nn.Module):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

        self.policy_model = PolicyModel()

    def forward(self):
        reward, debug_info = self.play_game()

        return reward, debug_info
    
    def play_game(self):

        state = State()

        for round_idx in range(13):
            # set round index
            state.round_index = round_idx
            self.play_round(state)

    def play_round(self, state: State):
        state = self.roll_dice(state)

        for roll_idx in range(2):
            state.rolls_remaining = 2 - roll_idx
            a = self.policy_model(state.get_feature_vector())
            state = self.roll_dice(state, a.dice_action)
            
            
        a = self.category_policy_model(state.get_feature_vector())
        state = self.select_categories(state, a.category_action)
    
    def roll_dice(self, state: State, a: Action | None):
        raise NotImplementedError