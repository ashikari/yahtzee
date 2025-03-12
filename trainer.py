import torch

from game_model import Yahtzee
from tqdm import tqdm
import time

import argparse


class Trainer:
    def __init__(self, batch_size: int, num_steps: int, log_interval: int):
        self.model = Yahtzee(batch_size=batch_size)
        self.num_steps = num_steps
        self.log_interval = log_interval
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.progress_bar = tqdm(
            range(self.num_steps),
            desc="Training Yahtzee",
            unit="step",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
        )

    def train(self):
        self.start_time = time.time()

        for _ in self.progress_bar:
            self.model()
            # rewards, actions = self.model()
            # loss = self.compute_loss(rewards, actions)
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()

            # if step % self.log_interval == 0:
            #     print(f"Step {step} completed")
            self.log()

    def compute_loss(self, rewards, actions):
        raise NotImplementedError

    def log(
        self,
        # loss
    ):
        # Update progress bar with metrics
        elapsed = time.time() - self.start_time
        self.progress_bar.set_postfix(
            {
                "batch": self.model.batch_size,
                "elapsed": f"{elapsed:.2f}s",
                # 'loss': f'{loss.item():.4f}' if 'loss' in locals() else 'N/A',
                # 'avg_reward': f'{rewards.mean().item():.2f}' if 'rewards' in locals() else 'N/A'
            }
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    trainer = Trainer(
        batch_size=args.batch_size, num_steps=args.num_steps, log_interval=10
    )
    trainer.train()
