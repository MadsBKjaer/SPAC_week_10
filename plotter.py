from matplotlib import pyplot as plt
from os import path


class Plotter:
    def __init__(self, plot: bool = True):
        self.plot = plot
        if plot:
            plt.ion()
        self.x = []
        self.y = []
        self.graph = plt.plot(self.x, self.y)[0]
        self.max_x = 0

    def update(self, x, reward):
        self.x.append(x)
        self.y.append(reward)

        self.graph.remove()
        self.graph = plt.plot(self.x, self.y, color="blue")[0]
        plt.title(f"{x + 1:6} - {(x + 1)/self.max_x:6.1%}")
        plt.pause(0.00001)

    def save_figure(self, name: str) -> None:
        plt.savefig(path.join("fig", f"{name}.png"))
