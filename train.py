import torch
from typing import Any
from torch.utils.data import DataLoader
from os import path


class Backpropagation:
    def __init__(
        self,
        model,
        model_name: str,
        loss_function,
        optimizer,
        scheduler=None,
        optimizer_kwargs: dict[str, Any] = {},
        scheduler_kwargs: dict[str, Any] = {},
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.loss_function = loss_function()
        self.optimizer = optimizer(model.parameters(), **optimizer_kwargs)
        self.scheduler = (
            scheduler(self.optimizer, **scheduler_kwargs)
            if scheduler is not None
            else None
        )

    def update_step(self, loss) -> None:
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step(loss)

    def train(
        self,
        dataset,
        batch_size: int,
        epochs: int,
        dataloader_kwargs: dict[str, Any] = {},
    ):
        self.model.train()
        dataloader = DataLoader(dataset, batch_size=batch_size, **dataloader_kwargs)
        best_loss = torch.inf
        best_accuracy = 0
        for i in range(epochs):
            for batch_nr, batch in enumerate(dataloader):
                prediction = self.model(batch)
                loss = self.loss_function(prediction, batch.reshape_as(prediction))
                self.update_step(loss)
                new_loss = loss.item()
            if best_loss > new_loss:
                new_accuracy = self.evaluate(dataset, batch_size)
                if best_accuracy < new_accuracy:
                    best_loss = new_loss
                    best_accuracy = new_accuracy
                    torch.save(
                        self.model.state_dict(),
                        path.join("models", f"{self.model_name}.pt"),
                    )
                    print(i, f"{best_accuracy:.3e}, {best_accuracy:.2%}")
            if best_accuracy > 0.999:
                print("Target accuracy reached")
                break

    def evaluate(
        self, dataset, batch_size: int, dataloader_kwargs: dict[str, Any] = {}
    ):
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, **dataloader_kwargs)
        for batch in dataloader:
            prediction = self.model(batch)
            argmax_batch = torch.reshape(batch, (batch.size()[0], 5, 26)).max(-1)[1]
            argmax_pred = torch.reshape(prediction, (batch.size()[0], 5, 26)).max(-1)[1]
        return torch.eq(argmax_batch, argmax_pred).to(torch.float16).mean().item()
