from matplotlib import pyplot as plt
from os import path


class Plotter:
    def __init__(self, plot: bool = True):
        self.plot = plot
        if plot:
            plt.ion()
        self.fig, self.ax = plt.subplots()
        self.x = []
        self.bottom = []
        self.rewards = {
            "win": [],
            "correct_letter": [],
            "useful_letter": [],
            "useless_letter": [],
            "repeat_letter": [],
            "accepted_word": [],
        }

        self.graph = self.ax.bar([], [])

    def update(self, x, rewards):
        self.x.append(x)
        self.bottom.append(0)

        baseline = 0
        for key in ["accepted_word", "repeat_letter"]:
            self.rewards[key].append(rewards[key] + baseline)
            baseline += rewards[key]

        baseline = 0
        for key in ["useless_letter", "useful_letter", "correct_letter", "win"]:
            self.rewards[key].append(rewards[key] + baseline)
            baseline += rewards[key]

        self.graph.remove()
        for reward_tuple, color in zip(
            self.rewards.items(),
            ["purple", "blue", "green", "yellow", "red", "orange"],
        ):
            type, reward = reward_tuple
            self.graph = self.ax.bar(
                self.x, reward, width=1, bottom=self.bottom, label=type, color=color
            )[0]
        if len(self.bottom) == 1:
            plt.legend("sw")
        if self.plot:
            plt.pause(0.00001)

    def save_figure(self, name: str) -> None:
        plt.savefig(path.join("fig", f"{name}.png"))
