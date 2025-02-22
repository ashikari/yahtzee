import torch
from dataclasses import dataclass

from score import score
from policy_model import PolicyModel

@dataclass
class State:
    dice: torch.Tensor
    categories: torch.Tensor


class Yahtzee(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.policy_model = PolicyModel()

    def forward(self, state: State):

        state = self.get_initial_state()

        reward, debug_info = self.play_game(state)

        return reward, debug_info
    
    def update_state_round_idx(self, state, round_idx):
        raise NotImplementedError
    
    def update_state_roll_idx(self, state, roll_idx):
        raise NotImplementedError
    
    def play_game(self, initial_state: State):

        state = initial_state

        for round_idx in range(13):
            state = self.update_state_round_idx(state, round_idx)
            round_score = self.play_round(state)
            score += round_score

        return round_score

    def play_round(self, state: State):
        round_score = 0
        for roll_idx in range(3):
            state = self.update_state_roll_idx(state, roll_idx)
            a = self.policy_model(state)
            state = self.roll_dice(state, a)
            
            round_score += score(state)

        return round_score
    
    def get_initial_state(self):
        raise NotImplementedError
    
    def roll_dice(self, state: State):
        raise NotImplementedError