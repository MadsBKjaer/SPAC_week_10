import torch
from train import GRPO
from model import WordleBot
from wordle import torchWordle
from os import path
from plotter import Plotter


if __name__ == "__main__":
    data_path: str = path.join("data", "wordle_words_sorted.csv")

    embedding_size = 64
    retrain = False
    activation = torch.nn.ReLU()

    wordle = torchWordle(1000, data_path)
    wordle_bot = WordleBot(activation, embedding_size)
    grpo = GRPO(
        wordle_bot,
        "wordle_bot",
        wordle,
        0.1,
        torch.optim.Adam(wordle_bot.parameters()),
        Plotter(True),
        max_turns=10,
    )
    grpo.train(1000, 10)
