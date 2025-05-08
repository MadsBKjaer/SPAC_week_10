from string import ascii_lowercase
from os import path
import polars as pl
import torch
from tokenizer import tokenizer, to_encoding
from collections import defaultdict
import timeit


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


class torchWordle(Wordle):
    def __init__(self, parallel_games: int, accepted_words_path, samples=None):
        super().__init__(accepted_words_path, samples)
        self.parallel_games = parallel_games
        self.samples = len(self.sample_words)
        self.reset_game_states()

    def reset_game_states(self):
        self.target_words = tokenizer(
            self.sample_words.sample(
                self.parallel_games,
                with_replacement=self.samples < self.parallel_games,
            ).to_list()
        )
        self.useless_letters = torch.zeros((self.parallel_games, 26)).bool()
        self.useful_letters = torch.zeros((self.parallel_games, 26, 5)).bool()
        self.correct_letters = torch.zeros((self.parallel_games, 5)).bool()
        self.done = torch.zeros((self.parallel_games,)).bool()
        self.rewards = torch.zeros((self.parallel_games, 1))

    def guess_word(self, words: torch.Tensor):
        if words.size()[0] != self.parallel_games:
            words = words.expand_as(self.target_words[0])

        # Game logic
        self.correct_letters = torch.logical_or(
            self.correct_letters, torch.eq(words, self.target_words[0])
        )
        self.done = torch.logical_or(self.correct_letters.all(-1), self.done)
        encoding = to_encoding(words)
        self.useless_letters = torch.logical_or(
            self.useless_letters,
            torch.logical_and(
                encoding.any(-2), torch.logical_not(self.target_words[1]).any(-2)
            ),
        )
        useful_index = torch.logical_and(encoding.any(-2), self.target_words[1].any(-2))
        self.useful_letters[useful_index] = torch.logical_or(
            self.useful_letters[useful_index],
            torch.transpose(encoding, -1, -2)[useful_index],
        )

        # Reward logic

        return self.done

    def rewards(self) -> torch.Tensor:
        torch.cat([self.useless_letters, self.useful_letters, self.correct_letters])


def test_run(n):
    wordle = torchWordle(2**n, path.join("data", "wordle_words_sorted.csv"))

    words = ["which"]
    tokenized_words = tokenizer(words)
    wordle.guess_word(tokenized_words[0])
    words = ["those"]
    tokenized_words = tokenizer(words)
    wordle.guess_word(tokenized_words[0])
    words = ["would"]
    tokenized_words = tokenizer(words)
    wordle.guess_word(tokenized_words[0])
    words = ["facts"]
    tokenized_words = tokenizer(words)
    wordle.guess_word(tokenized_words[0])
    words = ["odder"]
    tokenized_words = tokenizer(words)
    wordle.guess_word(tokenized_words[0])


if __name__ == "__main__":
    # n = 0
    # time = 0
    # while time < 1:
    #     timer = timeit.Timer(lambda: test_run(n))
    #     time = timer.timeit(100) / 100
    #     print(n, time)
    #     n += 1
    # 2**18=262144 games can be played in parallel in under a second
    print(2**18)

    timer = timeit.Timer(lambda: test_run(2**18))
    time = timer.timeit(10) / 10
    print(time)
