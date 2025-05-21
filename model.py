import torch
import polars as pl
from string import ascii_lowercase


class Encoder(torch.nn.Module):
    def __init__(self, activation, embedding_size: int, bias: bool = True) -> None:
        super().__init__()
        self.activation = activation
        self.embedding_size = embedding_size
        self.en1 = torch.nn.Linear(26 * 5, embedding_size, bias)
        self.en2 = torch.nn.Linear(embedding_size + 26 * 5, embedding_size, bias)
        self.en3 = torch.nn.Linear(embedding_size + 26 * 5, embedding_size, bias)
        self.en4 = torch.nn.Linear(embedding_size + 26 * 5, embedding_size, bias)

    def forward(self, input: torch.Tensor):
        input = torch.atleast_2d(input)
        x1 = self.en1(input)
        x1 = self.activation(x1)
        x2 = self.en2(torch.cat([input, x1], -1))
        x2 = self.activation(x2)
        x3 = self.en3(torch.cat([input, x2], -1))
        x3 = self.activation(x3)
        return torch.stack([x1, x2, x3], -1).mean(-1)


class Decoder(torch.nn.Module):
    def __init__(self, activation, embedding_size: int) -> None:
        super().__init__()
        self.activation = activation
        self.embedding_size = embedding_size
        self.de1 = torch.nn.Linear(embedding_size, 26 * 5)
        self.de2 = torch.nn.Linear(embedding_size + 26 * 5, 26 * 5)
        self.de3 = torch.nn.Linear(embedding_size + 26 * 5, 26 * 5)
        self.de4 = torch.nn.Linear(embedding_size + 26 * 5, 26 * 5)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x1 = self.de1(input)
        x1 = self.activation(x1)
        x2 = self.de2(torch.cat([input, x1], -1))
        x2 = self.activation(x2)
        x3 = self.de3(torch.cat([input, x2], -1))
        x3 = self.activation(x3)
        output = torch.stack([x1, x2, x3], -1).mean(-1)
        output = torch.reshape(output, (output.size()[0], 5, 26))
        return output


class AutoEncoder(torch.nn.Module):
    def __init__(self, activation, embedding_size: int) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.Encoder = Encoder(activation, embedding_size)
        self.Decoder = Decoder(activation, embedding_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.Encoder(input)
        return self.Decoder(x).flatten(1)


class WordleBot(torch.nn.Module):
    def __init__(
        self, activation, embedding_size: int, accepted_words_path: str
    ) -> None:
        super().__init__()
        self.activation = activation
        self.embedding_size = embedding_size
        self.rec1 = torch.nn.Linear(embedding_size * 2, embedding_size)
        self.rec2 = torch.nn.Linear(embedding_size * 2, embedding_size)
        self.rec3 = torch.nn.Linear(embedding_size * 2, embedding_size)
        self.gs1 = torch.nn.Linear(26 * 6 + 5, embedding_size)
        self.gs2 = torch.nn.Linear(26 * 6 + 5 + embedding_size, embedding_size)
        self.decoder = Decoder(activation, embedding_size)
        self.accepted_words: pl.LazyFrame = pl.scan_csv(accepted_words_path).drop(
            "occurrence"
        )
        self.accepted_words = (
            self.accepted_words.with_columns([pl.col("word").str.split("")])
            .with_columns(
                pl.col("word").list.get(0).alias("0"),
                pl.col("word").list.get(1).alias("1"),
                pl.col("word").list.get(2).alias("2"),
                pl.col("word").list.get(3).alias("3"),
                pl.col("word").list.get(4).alias("4"),
            )
            .drop("word")
        ).with_columns(
            pl.all().replace_strict(
                list(ascii_lowercase), list(range(len(ascii_lowercase)))
            )
        )

    def encode_game_state(self, game_state) -> torch.Tensor:
        x1 = self.activation(self.gs1(game_state))
        x2 = self.activation(self.gs2(torch.cat([game_state, x1], -1)))
        return torch.atleast_2d(torch.stack([x1, x2], -1).mean(-1))

    def recurrent_block(
        self, game_state: torch.Tensor, internal_state: torch.Tensor
    ) -> torch.Tensor:
        x1 = self.rec1(torch.cat([game_state, internal_state], -1))
        x1 = self.activation(x1)
        x2 = self.rec2(torch.cat([x1, internal_state], -1))
        x2 = self.activation(x2)
        x3 = self.rec3(torch.cat([x2, internal_state], -1))
        x3 = self.activation(x3)
        return torch.stack([x1, x2, x3], -1).mean(-1)

    def forward(
        self,
        game_state: torch.Tensor,
        max_recurrence: int = 1,
    ) -> torch.Tensor:
        game_state = self.encode_game_state(game_state)
        internal_state1 = torch.zeros((game_state.size()[0], self.embedding_size))
        internal_state2 = self.recurrent_block(game_state, internal_state1)
        # internal_state3 = self.recurrent_block(game_state, internal_state2)
        return self.decoder(torch.nn.functional.normalize(internal_state2))

    def sample_word(self, logits: torch.Tensor) -> torch.Tensor:
        softmax = torch.nn.Softmax(-1)
        base_prob = softmax(logits)
        sampling_order = torch.multinomial(base_prob.flatten(end_dim=1), 26).reshape_as(
            base_prob
        )
        batch_size = logits.size(0)
        batch_range = torch.arange(batch_size)
        word = torch.zeros((batch_size, 5), dtype=torch.int64) - 1
        word_bool = torch.ones((batch_size, 5), dtype=torch.bool)
        filter = [self.accepted_words for _ in batch_range]

        while True:
            masked_positions = torch.arange(5).expand((batch_size, 5))[word_bool]
            n_positions = int((masked_positions.size(-1) / batch_size))
            masked_positions = masked_positions.unfold(-1, n_positions, n_positions)

            masked_index_handler = (
                torch.ones((batch_size, n_positions, 26), dtype=torch.int64) * 100
            )
            masked_sampling_order = sampling_order[word_bool].reshape(
                batch_size, n_positions, 26
            )
            valid_letter_mask = (masked_sampling_order + 1).nonzero(as_tuple=True)
            masked_index_handler[valid_letter_mask] = torch.arange(26).expand_as(
                masked_index_handler
            )[valid_letter_mask]
            index_first_valid_letter = masked_index_handler.min(-1)[0]
            first_valid_letter = masked_sampling_order[
                batch_range,
                torch.arange(n_positions).expand((batch_size, n_positions)).T,
                index_first_valid_letter.T,
            ].T
            valid_sample_logits = logits[
                batch_range,
                masked_positions.T,
                first_valid_letter.T,
            ].T
            position_probability = softmax(valid_sample_logits)
            position = torch.multinomial(position_probability, 1)
            final_letter = first_valid_letter[batch_range, position.T].T
            word[batch_range, masked_positions[batch_range, position.T]] = (
                final_letter.T
            )
            word_bool[batch_range, masked_positions[batch_range, position.T]] = False

            if not word_bool.any():
                break

            for batch in batch_range:
                locked_position = masked_positions[batch, position[batch]].item()
                filter[batch] = (
                    filter[batch].filter(
                        pl.col(f"{locked_position}") == final_letter[batch].item()
                    )
                    # .drop(f"{locked_position}")
                )
                print(f"Batch {batch.item()}: Position {locked_position}")
                print(filter[batch].unique().collect())
                for pos in masked_positions[batch]:
                    print(sampling_order[batch, pos])
                    sampling_order[
                        batch,
                        pos,
                        sampling_order[batch, pos]
                        not in filter[batch].select(f"{pos}").collect(),
                    ] = -1
        return word
