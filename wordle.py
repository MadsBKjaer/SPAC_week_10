from string import ascii_lowercase
from os import path
import polars as pl
import torch

# alphabet: set[str] = set(ascii_lowercase)
# vowels: set[str] = set("aeiouy")
# consonants: set[str] = alphabet - vowels


class Wordle:
    accepted_words: pl.Series
    sample_words: pl.Series
    max_guesses: int
    guesses: int

    def __init__(
        self,
        accepted_words_path: str,
        sample_head: int | None = None,
        max_guesses: int = 5,
    ):
        self.accepted_words = pl.read_csv(accepted_words_path)["word"]
        self.sample_words = (
            self.accepted_words[:sample_head]
            if sample_head is not None
            else self.accepted_words
        )
        self.max_guesses = max_guesses
        self.reset_game_state()

    def reset_game_state(self) -> None:
        self.game_state_letter = torch.zeros((2, 26))
        self.game_state_placement = torch.zeros((5,))
        self.target_word = self.sample_words.sample()[0]
        self.guesses = 0

    def guess_word(self, word: str) -> bool:
        if word == self.target_word:
            return True

        _word: set[str] = set(word)
        _target_word: set[str] = set(self.target_word)

        for letter in _word & _target_word:
            self.game_state_letter[0, ascii_lowercase.index(letter)] = 1

        for letter in _word - _target_word:
            self.game_state_letter[1, ascii_lowercase.index(letter)] = 1

        for i, letter, target_letter in zip(range(5), word, self.target_word):
            if letter == target_letter:
                self.game_state_placement[i] = 1

        return False


if __name__ == "__main__":
    wordle = Wordle(path.join("data", "wordle_words_sorted.csv"), 100)
    wordle.target_word = "about"
    wordle.guess_word("those")
    print(wordle.game_state_letter)
    print(wordle.game_state_placement)
