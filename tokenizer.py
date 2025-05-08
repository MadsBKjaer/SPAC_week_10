import torch
from string import ascii_lowercase


def to_index(words: list[str]) -> torch.Tensor:
    """
    output needs a .flatten(0, 1) before using multinomial, which can be followed up with a reshape(len(words), 5) to receive expected size
    """
    assert all(map(lambda word: len(word) == 5, words)), "Not all words are of length 5"

    indexes = torch.tensor(
        list(map(lambda word: list(map(ascii_lowercase.index, word)), words))
    )

    return indexes


def to_encoding(indexes: torch.Tensor) -> torch.Tensor:
    n_words = indexes.size()[0]
    encoding = torch.zeros((n_words, 5, 26))
    encoding[
        torch.arange(n_words),
        torch.arange(5).unsqueeze(-1).expand(-1, n_words),
        indexes.T,
    ] = 1
    return encoding


def tokenizer(words: list[str]) -> torch.Tensor:
    indexes = to_index(words)
    return indexes, to_encoding(indexes)


if __name__ == "__main__":
    words: list[str] = ["trust", "which"]
    indexes = to_index(words)
    encoding = to_encoding(indexes)
    sample_multinomial = torch.multinomial(encoding.flatten(0, 1), 1)
    print(indexes)
    print(sample_multinomial.reshape((len(words), 5)))
