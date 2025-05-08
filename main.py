import torch
from train import Backpropagation, GRPO
from model import AutoEncoder, WordleBot
from tokenizer import tokenizer
from data_prep import WordleData
from wordle import Wordle
from os import path
from plotter import Plotter


if __name__ == "__main__":
    data_path: str = path.join("data", "wordle_words_sorted.csv")

    embedding_size = 20
    retrain = False
    activation = torch.nn.ReLU()

    autoencoder = AutoEncoder(activation, embedding_size, bias=True)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01, weight_decay=1e-8)
    training = Backpropagation(
        autoencoder,
        "autoencoder_word",
        torch.nn.MSELoss,
        optimizer,
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.9,
            patience=10,
            min_lr=1e-12,
            cooldown=10,
        ),
    )
    autoencoder_path = path.join("models", "autoencoder_word.pt")
    if not path.exists(autoencoder_path) or retrain:
        training.train(
            WordleData(data_path, tokenizer),
            batch_size=2**8,
            target_accuracy=0.99,
            max_epochs=100,
            dataloader_kwargs={"shuffle": True},
        )
    else:
        autoencoder.load_state_dict(torch.load(autoencoder_path))

    wordle = Wordle(data_path, 2**3)
    wordle_bot = WordleBot(activation, autoencoder, embedding_size)
    grpo = GRPO(
        wordle_bot,
        "wordle_bot",
        wordle,
        0.5,
        torch.optim.Adam(wordle_bot.parameters(), weight_decay=1e-8),
        Plotter(False),
        max_turns=5,
    )
    grpo.train(2**1, 8)
