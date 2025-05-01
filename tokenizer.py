import torch
from string import ascii_lowercase


def tokenizer(word: str) -> torch.Tensor:
    assert len(word) == 5, "Word should of length 5"
    word = word.lower()
    output = torch.zeros((5, 26))
    for i, letter in enumerate(word):
        output[i, ascii_lowercase.index(letter)] = 1
    return output.flatten()


if __name__ == "__main__":
    print(tokenizer("trust"))
