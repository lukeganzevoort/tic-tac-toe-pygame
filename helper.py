import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores, mean_scores, title: str):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title(title)
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    # plt.ylim(ymin=-100)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.1)


class PlotProgress:
    def __init__(self, title: str, plt_id: int):
        self.plot_scores: list[int] = []
        self.plot_mean_scores: list[float] = []
        self.total_score: int = 0
        self.n_games: int = 0
        self.title = title
        self.plt_id: int = plt_id

    def add_score(self, score: int):
        self.n_games += 1
        self.plot_scores.append(score)
        self.total_score += score
        mean_score = self.total_score / self.n_games
        self.plot_mean_scores.append(mean_score)
        plt.figure(self.plt_id)
        plot(self.plot_scores, self.plot_mean_scores, self.title)
