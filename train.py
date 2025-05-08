import torch
from typing import Any
from torch.utils.data import DataLoader
from os import path
from string import ascii_lowercase
from wordle import Wordle
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
        wordle: Wordle,
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
        log_probs = []
        game_states = []
        chosen_actions = []
        reward = defaultdict(lambda: 0, {})

        for turn in range(self.max_turns):
            game_state = self.wordle.game_state()
            game_states.append(game_state)
            probabilities = self.model(game_state).squeeze()
            try:
                action = torch.multinomial(probabilities, 1).squeeze()
                print(action)
            except Exception as e:
                print(game_state, probabilities, e)
            word = "".join(ascii_lowercase[i] for i in action.tolist())
            done, _reward = self.wordle.guess_word(word)

            for key, value in _reward.items():
                reward[key] += value

            log_prob = torch.log(probabilities[torch.arange(5), action]).squeeze()
            log_probs.append(log_prob.detach())
            chosen_actions.append(action)

            if done:
                break
        return game_states, log_probs, chosen_actions, reward

    def grpo_update(
        self,
        group_game_states,
        group_probs,
        group_actions,
        group_rewards,
        batch_size: int,
        n_iterations=20,
    ):

        advantages = (group_rewards - np.mean(group_rewards)) / (
            np.std(group_rewards) + 1e-8
        )

        for i_iter in range(n_iterations):
            loss = 0
            # iterating over each trajectory in the group
            for batch in range(batch_size):
                trajectory_loss = 0
                # iterating over each time step in the trajectory
                for t in range(len(group_game_states[batch])):
                    new_policy_probs = self.model(group_game_states[batch][t]).squeeze()
                    new_log_probs = torch.log(new_policy_probs)[
                        torch.arange(5), group_actions[batch][t]
                    ]

                    ratio = torch.exp(new_log_probs - group_probs[batch][t])
                    clipped_ratio = torch.clamp(
                        ratio, min=1 - self.eps, max=1 + self.eps
                    )
                    trajectory_loss = (
                        trajectory_loss - clipped_ratio * advantages[batch].item()
                    )
                trajectory_loss /= len(group_game_states[batch])
                loss = loss + trajectory_loss
            loss = (loss / batch_size).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, epochs: int, batch_size: int):
        for i_episode in range(epochs):
            print(f"{i_episode}/{epochs} = {i_episode/epochs:.1%}")
            group_game_states = []
            group_probs = []
            group_actions = []
            group_rewards = []
            _reward = defaultdict(lambda: 0, {})

            for batch in range(batch_size):
                game_states, log_probs, chosen_actions, reward = (
                    self.collect_trajectory()
                )

                for key, value in reward.items():
                    _reward[key] += value

                group_game_states.append(game_states)
                group_probs.append(log_probs)
                group_actions.append(chosen_actions)
                group_rewards.append(np.sum(list(reward.values())))

            # update policy using grpo on the collected trajectories
            self.grpo_update(
                group_game_states,
                group_probs,
                group_actions,
                group_rewards,
                batch_size,
                8,
            )

            # if i_episode % 1 == 0:
            #     print(np.mean(group_rewards))
            self.plot.update(i_episode, _reward)
        self.plot.save_figure(f"wordle_{epochs}_{batch_size}_{self.eps}")
