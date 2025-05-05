from string import ascii_lowercase
from os import path
import polars as pl
import torch
from collections import defaultdict


class Wordle:
    accepted_words: pl.Series
    sample_words: pl.Series
    max_guesses: int
    guesses: int

    def __init__(
        self,
        accepted_words_path: str,
        sample_head: int | None = None,
    ):
        self.accepted_words = pl.read_csv(accepted_words_path)["word"]
        self.sample_words = (
            self.accepted_words[:sample_head]
            if sample_head is not None
            else self.accepted_words
        )
        self.reset_game_state()

    def reset_game_state(self) -> None:
        self.game_state_letter = torch.zeros((2, 26))
        self.game_state_placement = torch.zeros((5,))
        self.target_word = self.sample_words.sample()[0]
        self.guesses = 0
        self.useful_letters: set[str] = set()
        self.useless_letters: set[str] = set()
        self.correct_letters: dict[int, str] = {}

    def game_state(self) -> torch.Tensor:
        return torch.cat(
            [self.game_state_letter.flatten(), self.game_state_placement], -1
        )

    def guess_word(self, word: str) -> tuple[bool, int]:
        self.guesses += 1
        _word: set[str] = set(word)
        _target_word: set[str] = set(self.target_word)
        useful_letters = _word & _target_word
        useless_letters = _word - _target_word
        reward = defaultdict(lambda: 0, {})
        if word not in self.accepted_words:
            #     return False, -1
            # print("guessed a proper word", word)
            reward["accepted_word"] += -1

        for letter in useful_letters - self.useful_letters:
            self.game_state_letter[0, ascii_lowercase.index(letter)] = 1
            reward["useful_letter"] += 3
        self.useful_letters = useful_letters | self.useful_letters

        for letter in _word & self.useless_letters:
            reward["repeat_letter"] += -5

        for letter in useless_letters - self.useless_letters:
            self.game_state_letter[1, ascii_lowercase.index(letter)] = 1
            reward["useless_letter"] += 2

        self.useless_letters = useless_letters | self.useless_letters

        for i, letters in enumerate(zip(word, self.target_word)):
            verbose = False
            if letters[0] == letters[1] and i not in self.correct_letters:
                self.game_state_placement[i] = 1
                self.correct_letters[i] = letters[0]

                reward["correct_letter"] += 5
                verbose |= False

        if verbose:
            print(word, self.correct_letters)

        if word == self.target_word == 5:
            print("Win", self.target_word)
            reward["win"] += 5
            return True, reward
        return False, reward


if __name__ == "__main__":
    wordle = Wordle(path.join("data", "wordle_words_sorted.csv"), 100)
    wordle.target_word = "about"
    wordle.guess_word("those")
    print(wordle.game_state_letter)
    print(wordle.game_state_placement)
