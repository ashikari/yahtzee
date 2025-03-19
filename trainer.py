import torch

from game_model import Yahtzee
from tqdm import tqdm
import time

from state import State
from typing import List

import argparse

import wandb

wandb.login()


class Trainer:
    def __init__(
        self, batch_size: int, num_steps: int, log_interval: int, use_gpu: bool = False
    ):
        device = torch.device(
            "mps:0" if torch.backends.mps.is_available() and use_gpu else "cpu"
        )
        self.model = Yahtzee(batch_size=batch_size, device=device)
        if device == torch.device("mps:0"):
            print("Compiling model")
            self.model = torch.compile(self.model, backend="aot_eager")

        self.num_steps = num_steps
        self.log_interval = log_interval
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        self.progress_bar = tqdm(
            range(self.num_steps),
            desc="Training Yahtzee",
            unit="step",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
        )

    def train(self):
        wandb.watch(self.model, log="all", log_freq=self.log_interval)
        self.start_time = time.time()

        for step in self.progress_bar:
            states, actions, rewards = self.model()
            loss = self.compute_loss(rewards, actions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % self.log_interval == 0:
                self.log(step, states, loss, rewards)

    def compute_loss(self, rewards, actions):
        device = rewards.device
        # Extract action log probabilities and their indices
        action_log_probs, action_indices = self._extract_action_log_probs(
            actions, device
        )

        # Get filtered rewards based on valid action indices
        filtered_rewards = self._get_filtered_rewards(rewards, action_indices)

        # Normalize rewards
        normalized_rewards = filtered_rewards - filtered_rewards.mean(dim=0)

        # Compute cumulative future rewards
        cumulative_rewards = self._compute_cumulative_rewards(normalized_rewards)

        # Calculate loss using policy gradient approach
        loss = -action_log_probs * cumulative_rewards.detach()
        return loss.sum(dim=1).mean()

    def _extract_action_log_probs(self, actions, device=torch.device("cpu")):
        """Extract log probabilities from valid actions and track their indices."""
        action_vector = []
        action_indices = []

        # Create tensors to store action types and their log probabilities
        for a_idx, action in enumerate(actions):
            # Skip actions with no log probabilities
            if (
                action.dice_action_log_prob is None
                and action.category_action_log_prob is None
            ):
                continue

            if action.dice_action_log_prob is not None:
                action_vector.append(
                    action.dice_action_log_prob.sum(dim=1, keepdim=True)
                )
            elif action.category_action_log_prob is not None:
                action_vector.append(action.category_action_log_prob.unsqueeze(1))

            action_indices.append(a_idx)

        return torch.cat(action_vector, dim=1), torch.tensor(
            action_indices, device=device
        )

    def _get_filtered_rewards(self, rewards, action_indices):
        """Filter rewards to only include those corresponding to valid actions."""
        return torch.index_select(rewards, dim=1, index=action_indices)

    def _compute_cumulative_rewards(self, rewards):
        """Compute cumulative future rewards for each timestep."""
        _, steps = rewards.shape
        indices = torch.arange(steps, device=rewards.device)

        # Create a mask where each element (i,j) is 1 if j >= i (lower triangular matrix)
        mask = indices.unsqueeze(0) <= indices.unsqueeze(1)

        # Matrix multiplication to efficiently compute cumulative future rewards
        return torch.matmul(rewards, mask.float())

    def log(
        self,
        step: int,
        states: List[State],
        loss: torch.Tensor,
        rewards: torch.Tensor,
    ):
        average_score = states[-1].total_score.mean(dim=0).item()
        median_score = states[-1].total_score.median(dim=0).values.item()
        average_reward = rewards.mean().item()
        max_reward = rewards.max().item()
        min_reward = rewards.min().item()
        loss = loss.item()

        # Update progress bar with metrics
        elapsed = time.time() - self.start_time

        log_dict = {
            "Average Total Score": average_score,
            "Median Total Score": median_score,
            "Average Reward": average_reward,
            "Max Reward": max_reward,
            "Min Reward": min_reward,
            "loss": loss,
        }

        wandb.log(log_dict, step=step)
        local_log_dict = {
            "batch": self.model.batch_size,
            "elapsed": f"{elapsed:.2f}s",
        }
        local_log_dict.update(log_dict)

        self.progress_bar.set_postfix(local_log_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU for training if available"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with wandb.init(project="yahtzee-rl", config=args):
        config = wandb.config

        if config.seed is not None:
            torch.manual_seed(config.seed)

        trainer = Trainer(
            batch_size=config.batch_size,
            num_steps=config.num_steps,
            log_interval=25,
            use_gpu=config.use_gpu,
        )
        trainer.train()
