from abc import abstractmethod

from torch.utils.tensorboard.writer import SummaryWriter
import itertools as it


class Plot:
    # def __init__(self, *, training: bool, instance_id: int, episode: int):
    def __init__(self, **kwargs):

        self.active = False

        for k, v in kwargs.items():
            setattr(self, k, v)

        # self.training = training
        # self.instance_id = instance_id
        # self.episode = episode

        self.counter = it.count(0)

        self.writer = SummaryWriter()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.plot()

    @abstractmethod
    def plot(self):
        pass
