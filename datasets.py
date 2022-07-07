from abc import ABC
from typing import Any, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class Connect4(Dataset, ABC):
    convert_data = dict(zip(*pd.factorize(['b', 'x', 'o'], sort=True)))
    classes = dict(zip(*pd.factorize(['win', 'loss', 'draw'], sort=True)))

    def __init__(self, root: str, train: bool = True) -> None:
        super(Connect4, self).__init__()

        self.num_classes = len(self.classes)
        self.train = train

        data = pd.read_csv(f'{root}/Connect-4/connect-4.data', sep=',', header=None)
        data = data.apply(lambda col: pd.factorize(col, sort=True)[0]).to_numpy()
        size = data.shape[0]
        cut = int(0.9 * size)

        if self.train:
            self.data = data[:cut, :-1]
            self.targets = data[:cut, -1]
        else:
            self.data = data[cut:, :-1]
            self.targets = data[cut:, -1]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        vector = torch.tensor(self.data[index])
        target = torch.tensor(self.targets[index])
        return vector.float(), target

    def __len__(self):
        return self.targets.size


class Letter(Dataset, ABC):

    classes = dict(zip(*pd.factorize(
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
         'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
        sort=True)))

    def __init__(self, root: str, train: bool = True) -> None:
        super(Letter, self).__init__()

        self.num_classes = len(self.classes)
        self.train = train

        data = pd.read_csv(f'{root}/Letter/letter.data', sep=',', header=None)
        data = data.apply(lambda col: pd.factorize(col, sort=True)[0]).to_numpy()
        size = data.shape[0]
        cut = int(0.9 * size)

        if self.train:
            self.data = data[:cut, :-1]
            self.targets = data[:cut, -1]
        else:
            self.data = data[cut:, :-1]
            self.targets = data[cut:, -1]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        vector = torch.tensor(self.data[index])
        target = torch.tensor(self.targets[index])
        return vector.float(), target

    def __len__(self):
        return self.targets.size
