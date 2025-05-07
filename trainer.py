import torch

from game_model import Yahtzee
from tqdm import tqdm
import time

from state import State
from typing import List

import argparse

import wandb

from dataclasses import dataclass

wandb.login()


@dataclass
class Loss:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor = None


class Trainer:
    def __init__(
        self,
        batch_size: int,
        num_steps: int,
        log_interval: int,
        use_gpu: bool = False,
        initial_lr: float = 0.001,
        lr_schedule: str = "constant",
        decay_rate: float = 0.9,
        step_size: int = 1000,
        policy_loss_coefficient: float = 100.0,
        value_loss_coefficient: float = 0.01,
        entropy_loss_coefficient: float = 1,
        use_learned_value: bool = False,
        a2c: bool = False,
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
        self.initial_lr = initial_lr
        self.lr_schedule = lr_schedule
        self.policy_loss_coefficient = policy_loss_coefficient
        self.value_loss_coefficient = value_loss_coefficient
        self.entropy_loss_coefficient = entropy_loss_coefficient
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr)
        self.use_learned_value = use_learned_value
        self.a2c = a2c
        # Set up learning rate scheduler based on the chosen schedule
        if lr_schedule == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=decay_rate
            )
        elif lr_schedule == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=decay_rate
            )
        else:  # "constant" or any other value
            self.scheduler = None

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
            states, actions, values, rewards = self.model()
            loss = self.compute_loss(rewards, actions, values)
            self.optimizer.zero_grad()
            loss.total_loss.backward()
            self.optimizer.step()

            # Update learning rate according to schedule
            if self.scheduler is not None:
                self.scheduler.step()

            if step % self.log_interval == 0:
                self.log(step, states, values, loss, rewards)

    def compute_loss(self, rewards, actions, values):
        device = rewards.device
        # Extract action log probabilities and their indices
        action_log_probs, action_entropy, action_indices = (
            self._extract_action_probs_and_entropy(actions, device)
        )

        # Get filtered rewards based on valid action indices
        filtered_rewards = self._get_filtered_rewards(rewards, action_indices)

        # Compute cumulative future rewards
        # shape: (batch_size, num_steps)
        cumulative_rewards = self._compute_cumulative_rewards(filtered_rewards)

        # Value loss calculation using PyTorch's built-in Huber loss
        huber_loss = torch.nn.HuberLoss(reduction="none")
        value_loss = huber_loss(values, cumulative_rewards.detach())
        value_loss = value_loss.sum(dim=1).mean()

        # Use advantages for policy loss
        if self.use_learned_value:
            advantages = cumulative_rewards.detach() - values.detach()
            # normalize advantages
            advantages = (advantages - advantages.mean(dim=0)) / advantages.std(dim=0)
            policy_loss = -action_log_probs * advantages.detach()
        elif self.a2c:
            # set the last value of s_t+1 to 0
            next_tick_values = torch.zeros_like(values)
            next_tick_values[:, :-1] = values[:, 1:].detach()
            advantages = filtered_rewards.detach() + next_tick_values - values.detach()
            # normalize advantages
            advantages = (advantages - advantages.mean(dim=0)) / advantages.std(dim=0)
            policy_loss = -action_log_probs * advantages.detach()
        else:
            policy_loss = -action_log_probs * (
                cumulative_rewards.detach() - cumulative_rewards.mean(dim=0).detach()
            )
        policy_loss = policy_loss.sum(dim=1).mean()

        # Add entropy regularization
        entropy_loss = -action_entropy
        entropy_loss = entropy_loss.sum(dim=1).mean()

        total_loss = (
            self.policy_loss_coefficient * policy_loss
            + self.value_loss_coefficient * value_loss
            + self.entropy_loss_coefficient * entropy_loss
        )

        return Loss(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
        )

    def _extract_action_probs_and_entropy(self, actions, device=torch.device("cpu")):
        """Extract log probabilities from valid actions and track their indices."""
        # TODO: Make this function more efficient by pre-computing the action indices
        action_vector = []
        action_entropy = []
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
                action_entropy.append(
                    action.dice_action_entropy.sum(dim=1, keepdim=True)
                )
            elif action.category_action_log_prob is not None:
                action_vector.append(action.category_action_log_prob.unsqueeze(1))
                action_entropy.append(action.category_action_entropy.unsqueeze(1))

            action_indices.append(a_idx)

        return (
            torch.cat(action_vector, dim=1),
            torch.cat(action_entropy, dim=1),
            torch.tensor(action_indices, device=device),
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
        values: torch.Tensor,
        loss: Loss,
        rewards: torch.Tensor,
    ):
        average_score = states[-1].total_score.mean(dim=0).item()
        median_score = states[-1].total_score.median(dim=0).values.item()
        average_reward = rewards.mean().item()
        max_reward = rewards.max().item()
        min_reward = rewards.min().item()
        total_loss = loss.total_loss.item()
        policy_loss = loss.policy_loss.item()
        value_loss = loss.value_loss.item()
        entropy_loss = loss.entropy_loss.item()
        current_lr = self.optimizer.param_groups[0]["lr"]
        average_value = values.mean().item()

        # Update progress bar with metrics
        elapsed = time.time() - self.start_time

        log_dict = {
            "Average Total Score": average_score,
            "Median Total Score": median_score,
            "Average Reward": average_reward,
            "Max Reward": max_reward,
            "Min Reward": min_reward,
            "Total Loss": total_loss,
            "Policy Loss": policy_loss,
            "Value Loss": value_loss,
            "Entropy Loss": entropy_loss,
            "learning_rate": current_lr,
            "Average Value": average_value,
        }

        wandb.log(log_dict, step=step)

        local_log_dict = {
            "elapsed": f"{elapsed:.2f}s",
            "average_reward": average_reward,
            "Average Total Score": average_score,
            "Median Total Score": median_score,
            "Total Loss": total_loss,
        }

        self.progress_bar.set_postfix(local_log_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU for training if available"
    )
    parser.add_argument(
        "--initial_lr", type=float, default=0.001, help="Initial learning rate"
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="constant",
        choices=["constant", "exponential", "step"],
        help="Learning rate schedule type",
    )
    parser.add_argument(
        "--decay_rate",
        type=float,
        default=0.9,
        help="Decay rate for learning rate schedules",
    )
    parser.add_argument(
        "--step_size", type=int, default=4000, help="Step size for step decay schedule"
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--policy_loss_coefficient",
        type=float,
        default=100.0,
        help="Coefficient for policy loss",
    )
    parser.add_argument(
        "--value_loss_coefficient",
        type=float,
        default=0.01,
        help="Coefficient for value loss",
    )
    parser.add_argument(
        "--entropy_loss_coefficient",
        type=float,
        default=1,
        help="Coefficient for entropy regularization",
    )
    parser.add_argument(
        "--use_learned_value",
        action="store_true",
        help="Use learned value estimation as a baseline instead of average Monte Carlo returns",
    )
    parser.add_argument(
        "--a2c",
        action="store_true",
        help="Use a single sample next step value estimate to compute the advantage",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with wandb.init(
        project="yahtzee-rl",
        config=args,
        mode="disabled" if args.disable_wandb else "online",
    ):
        config = wandb.config

        if config.seed is not None:
            torch.manual_seed(config.seed)

        trainer = Trainer(
            batch_size=config.batch_size,
            num_steps=config.num_steps,
            log_interval=25,
            use_gpu=config.use_gpu,
            initial_lr=config.initial_lr,
            lr_schedule=config.lr_schedule,
            decay_rate=config.decay_rate,
            step_size=config.step_size,
            entropy_loss_coefficient=config.entropy_loss_coefficient,
            use_learned_value=config.use_learned_value,
        )
        trainer.train()
