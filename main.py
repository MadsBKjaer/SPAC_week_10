import torch
from train import Backpropagation
from model import AutoEncoder
from tokenizer import tokenizer
from data_prep import WordleData
from os import path

if __name__ == "__main__":
    data_path: str = path.join("data", "wordle_words_sorted.csv")
    autoencoder = AutoEncoder(torch.nn.ELU(), bias=True)
    training = Backpropagation(
        autoencoder,
        "autoencoder_word",
        torch.nn.MSELoss,
        torch.optim.Adam,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        optimizer_kwargs={"lr": 0.01},
        scheduler_kwargs={
            "factor": 0.9,
            "patience": 10,
            "min_lr": 1e-16,
            "eps": 0,
            "cooldown": 10,
        },
    )

    training.evaluate(WordleData(data_path, tokenizer), batch_size=2**8)
    training.train(
        WordleData(data_path, tokenizer),
        batch_size=2**8,
        epochs=100,
        dataloader_kwargs={"shuffle": True},
    )
    training.evaluate(WordleData(data_path, tokenizer), batch_size=2**8)
