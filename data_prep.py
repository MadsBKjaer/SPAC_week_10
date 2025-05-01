import polars as pl
from os import path
from collections.abc import Callable
from torch.utils.data import Dataset
from tokenizer import tokenizer
import torch


def sort_csv(
    file_path: str, save_path: str, column: str, descending: bool = False
) -> None:
    ldf: pl.LazyFrame = pl.scan_csv(file_path)
    ldf.sort(column, descending=descending).sink_csv(save_path)


def store_tokenizations():
    pass


class WordleData(Dataset):
    def __init__(
        self,
        data_file: str,
        tokenizer: Callable[[str], torch.Tensor],
        column: str = "word",
    ) -> None:
        self.data = pl.read_csv(data_file)[column]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> torch.Tensor:
        word = self.data[index]
        # print(word)
        return self.tokenizer(word)


if __name__ == "__main__":
    file_path: str = path.join("data", "wordle_words_sorted.csv")
    target_path: str = path.join("data", "wordle_words_sorted.csv")
    if not path.exists(target_path):
        sort_csv(file_path, target_path, "occurrence", True)

    data = WordleData(target_path, tokenizer)
    print(data[3])
