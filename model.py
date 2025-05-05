import torch


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
    def __init__(self, activation, embedding_size: int, bias: bool = True) -> None:
        super().__init__()
        self.activation = activation
        self.embedding_size = embedding_size
        self.de1 = torch.nn.Linear(embedding_size, 26 * 5, bias)
        self.de2 = torch.nn.Linear(embedding_size + 26 * 5, 26 * 5, bias)
        self.de3 = torch.nn.Linear(embedding_size + 26 * 5, 26 * 5, bias)
        self.de4 = torch.nn.Linear(embedding_size + 26 * 5, 26 * 5, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x1 = self.de1(input)
        x1 = self.activation(x1)
        x2 = self.de2(torch.cat([input, x1], -1))
        x2 = self.activation(x2)
        x3 = self.de3(torch.cat([input, x2], -1))
        x3 = self.activation(x3)
        output = torch.stack([x1, x2, x3], -1).mean(-1)
        output = torch.reshape(output, (output.size()[0], 5, 26))
        return torch.nn.Softmax(-1)(output)


class AutoEncoder(torch.nn.Module):
    def __init__(self, activation, embedding_size: int, bias: bool = True) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.Encoder = Encoder(activation, embedding_size, bias)
        self.Decoder = Decoder(activation, embedding_size, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.Encoder(input)
        return self.Decoder(x).flatten(1)


class WordleBot(torch.nn.Module):
    def __init__(
        self, activation, auto_encoder: AutoEncoder, embedding_size: int
    ) -> None:
        super().__init__()
        self.activation = activation
        self.out = torch.nn.Linear(32, embedding_size)
        self.rec1 = torch.nn.Linear(32 + 16, 32)
        self.rec2 = torch.nn.Linear(32 * 2, 32)
        self.rec3 = torch.nn.Linear(32 * 2, 32)
        self.gs1 = torch.nn.Linear(26 * 2 + 5, 16)
        self.gs2 = torch.nn.Linear(26 * 2 + 5 + 16, 16)
        self.auto_encoder = auto_encoder

        # Freezes the auto encoder parameters
        # for param in auto_encoder.parameters():
        #     param.requires_grad = False

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
        internal_state1 = torch.zeros((game_state.size()[0], 32))
        internal_state2 = self.recurrent_block(game_state, internal_state1)
        internal_state3 = self.recurrent_block(game_state, internal_state2)
        word_embedding = torch.nn.functional.normalize(self.out(internal_state3))
        return self.auto_encoder.Decoder(word_embedding)
