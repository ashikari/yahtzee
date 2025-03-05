import torch

from game_model import Yahtzee


class Trainer:
    def __init__(self, batch_size: int, num_steps: int, log_interval: int):
        self.model = Yahtzee(batch_size=batch_size)
        self.num_steps = num_steps
        self.log_interval = log_interval

    def train(self):
        for step in range(self.num_steps):
            rewards, actions = self.model()
            loss = self.compute_loss(rewards, actions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % self.log_interval == 0:
                self.log(step, loss)

    def compute_loss(self, rewards, actions):
        raise NotImplementedError

    def log(self, step, loss):
        raise NotImplementedError


if __name__ == "__main__":
    trainer = Trainer(batch_size=10, num_steps=100, log_interval=10)
    trainer.train()
