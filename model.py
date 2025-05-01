import torch


class Encoder(torch.nn.Module):
    def __init__(self, activation, bias: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = activation
        self.en1 = torch.nn.Linear(26 * 5, 26, bias)
        self.en2 = torch.nn.Linear(26 * 6, 26, bias)
        self.en3 = torch.nn.Linear(26 * 6, 26, bias)
        self.en4 = torch.nn.Linear(26 * 6, 26, bias)

    def forward(self, input: torch.Tensor):
        input = torch.atleast_2d(input)
        x1 = self.en1(input)
        x1 = self.activation(x1)
        x2 = self.en2(torch.cat([input, x1], -1))
        x2 = self.activation(x2)
        x3 = self.en3(torch.cat([input, x2], -1))
        x3 = self.activation(x3)
        # x4 = self.en4(torch.cat([input, x3], -1))
        # x4 = self.activation(x4)
        return torch.stack([x1, x2, x3], -1).mean(-1)


class Decoder(torch.nn.Module):
    def __init__(self, activation, bias: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = activation
        self.de1 = torch.nn.Linear(26, 26 * 5, bias)
        self.de2 = torch.nn.Linear(26 * 6, 26 * 5, bias)
        self.de3 = torch.nn.Linear(26 * 6, 26 * 5, bias)
        self.de4 = torch.nn.Linear(26 * 6, 26 * 5, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x1 = self.de1(input)
        x1 = self.activation(x1)
        x2 = self.de2(torch.cat([input, x1], -1))
        x2 = self.activation(x2)
        x3 = self.de3(torch.cat([input, x2], -1))
        x3 = self.activation(x3)
        # x4 = self.de4(torch.cat([input, x3], -1))
        # x4 = self.activation(x4)
        output = torch.stack([x1, x2, x3], -1).mean(-1)
        output = torch.reshape(output, (output.size()[0], 5, 26))
        return torch.nn.Softmax(-1)(output).flatten(1)


class AutoEncoder(torch.nn.Module):
    def __init__(self, activation, bias: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.Encoder = Encoder(activation, bias)
        self.Decoder = Decoder(activation, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.Encoder(input)
        return self.Decoder(x)
