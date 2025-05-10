import torch
from typing import Any
from torch.utils.data import DataLoader
from os import path
from string import ascii_lowercase
from wordle import torchWordle
import numpy as np
from collections import defaultdict


class Backpropagation:
    def __init__(
        self,
        model,
        model_name: str,
        loss_function,
        optimizer,
        scheduler=None,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.loss_function = loss_function()
        self.optimizer = optimizer
        self.scheduler = scheduler

    def update_step(self, loss) -> None:
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step(loss)

    def train(
        self,
        dataset,
        batch_size: int,
        target_accuracy: float,
        max_epochs: int,
        dataloader_kwargs: dict[str, Any] = {},
    ):
        # self.model.train()
        dataloader = DataLoader(dataset, batch_size=batch_size, **dataloader_kwargs)
        best_loss = torch.inf
        best_accuracy = 0
        for i in range(max_epochs):
            for batch in dataloader:
                prediction = self.model(batch)
                loss = self.loss_function(prediction, batch)
                self.update_step(loss)
                new_loss = loss.item()
            if best_loss > new_loss:
                new_accuracy = self.evaluate(dataset, batch_size)
                if best_accuracy < new_accuracy:
                    best_loss = new_loss
                    best_accuracy = new_accuracy
                    torch.save(
                        self.model.state_dict(),
                        path.join("models", f"{self.model_name}.pt"),
                    )
                    print(i, f"{best_accuracy:.3e}, {best_accuracy:.2%}")
            if best_accuracy > target_accuracy:
                print("Target accuracy reached")
                break

    def evaluate(
        self, dataset, batch_size: int, dataloader_kwargs: dict[str, Any] = {}
    ):
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, **dataloader_kwargs)
        for batch in dataloader:
            prediction = self.model(batch)
            argmax_batch = torch.reshape(batch, (batch.size()[0], 5, 26)).max(-1)[1]
            argmax_pred = torch.reshape(prediction, (batch.size()[0], 5, 26)).max(-1)[1]
        return torch.eq(argmax_batch, argmax_pred).to(torch.float16).mean().item()


class GRPO:
    def __init__(
        self,
        model,
        model_name: str,
        wordle: torchWordle,
        eps: float,
        optimizer,
        plotter,
        max_turns: int = 5,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.wordle = wordle
        self.max_turns = max_turns
        self.eps = eps
        self.optimizer = optimizer
        self.plot = plotter

    def collect_trajectory(self):
        self.wordle.reset_game_state()
        game_states: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []

        for _ in range(self.max_turns):
            game_state = self.wordle.game_state()
            probabilities = self.model(game_state)
            action = torch.multinomial(probabilities.flatten(0, 1), 1).reshape(
                self.wordle.parallel_games, 5
            )
            done = self.wordle.guess_word(action)

            log_prob = torch.log(
                probabilities[
                    torch.arange(self.wordle.parallel_games),
                    torch.arange(5).expand((self.wordle.parallel_games, 5)).T,
                    action.T,
                ]
            ).T

            game_states.append(game_state)
            log_probs.append(log_prob.detach())
            actions.append(action)

            if done.all():
                break
        return (
            torch.stack(game_states, 1),
            torch.stack(log_probs, 1),
            torch.stack(actions, 1),
            self.wordle.rewards().float(),
        )

    def grpo_update(
        self,
        game_states: torch.Tensor,
        log_probs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        n_iterations: int = 20,
    ) -> float:
        game_length: torch.Tensor = torch.logical_not(
            game_states[:, :, -6:-1].all(-1)
        ).sum(-1)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        normalized_advantage = advantages / game_length

        for i_iter in range(n_iterations):
            loss = 0
            # print(f"{game_states.size() = }")
            new_probs = self.model(game_states.flatten(end_dim=1)).reshape(
                (self.wordle.parallel_games, self.max_turns, 5, 26)
            )
            # print(f"{new_probs.size() = }")
            # print(f"{log_probs.size() = }")
            new_log_probs = torch.log(
                new_probs[
                    torch.arange(self.wordle.parallel_games),
                    torch.arange(self.max_turns)
                    .expand(self.wordle.parallel_games, self.max_turns)
                    .T,
                    torch.arange(5)
                    .expand(self.wordle.parallel_games, self.max_turns, 5)
                    .T,
                    actions.T,
                ]
            ).T
            # print(f"{new_log_probs.size() = }")

            ratio = torch.exp(new_log_probs - log_probs)
            # print(f"{ratio.size() = }")

            clipped_ratio = -torch.clamp(ratio, min=1 - self.eps, max=1 + self.eps)
            # print(f"{clipped_ratio.size() = }")

            # print(f"{advantages.size() = }")
            trajectory_loss = clipped_ratio * normalized_advantage.reshape(
                (self.wordle.parallel_games, 1, 1)
            )
            # print(f"{trajectory_loss.size() = }")

            # iterating over each trajectory in the group
            loss = trajectory_loss.sum() / self.wordle.parallel_games
            # print(f"{loss = }")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return rewards.mean().item()

    def train(self, epochs: int, iterations: int, verbose: int = False):
        for epoch in range(epochs):

            game_states, log_probs, actions, rewards = self.collect_trajectory()

            # update policy using grpo on the collected trajectories
            mean_reward = self.grpo_update(
                game_states,
                log_probs,
                actions,
                rewards,
                iterations,
            )
            if verbose:
                print(f"{epoch}/{epochs} = {epoch/epochs:.1%} - {mean_reward = }")
            # if i_episode % 1 == 0:
            #     print(np.mean(group_rewards))
            if epoch % 10 == 0:
                self.plot.update(epoch, mean_reward)
        # self.plot.save_figure(f"wordle_{epochs}_{batch_size}_{self.eps}")
