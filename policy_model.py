import torch
from game_model import State

class PolicyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state: State):
        raise NotImplementedError