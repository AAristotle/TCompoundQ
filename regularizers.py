from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
from torch import nn

class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class wN3(Regularizer):
    def __init__(self, weight: float):
        super(wN3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            h, r, t = factor
            norm += 2.0 * torch.sum(h**3)
            norm += 2.0 * torch.sum(t**3)
            norm += 0.5 * torch.sum(r**3)
        return self.weight * norm / h.shape[0]


class Lambda3(Regularizer):
    def __init__(self, weight: float):
        super(Lambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        ddiff = factor[1:] - factor[:-1]
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


class Linear3(Regularizer):
    def __init__(self, weight: float):
        super(Linear3, self).__init__()
        self.weight = weight

    def forward(self, factor, W):
        rank = int(factor.shape[1] / 2)
        ddiff = factor[1:] - factor[:-1] - W.weight[:rank*2].t()
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


class Spiral3(Regularizer):
    def __init__(self, weight: float):
        super(Spiral3, self).__init__()
        self.weight = weight

    def forward(self, factor, time_phase):
        ddiff = factor[1:] - factor[:-1]
        ddiff_pahse = time_phase[1:] - time_phase[:-1]
        rank = int(ddiff.shape[1] / 2)
        rank1 = int(ddiff_pahse.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2 + ddiff_pahse[:, :rank1]**2 + ddiff_pahse[:, rank1:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


class Spiral3_tp(Regularizer):
    def __init__(self, weight: float):
        super(Spiral3_tp, self).__init__()
        self.weight = weight

    def forward(self, factor, time_phase):
        ddiff = factor[1:] - factor[:-1] 
        ddiff_pahse = time_phase[1:] - time_phase[:-1]
        rank = int(ddiff.shape[1] / 2)
        rank1 = int(ddiff_pahse.shape[1] / 2)
        rank = rank1
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:2*rank]**2 + ddiff_pahse[:, :rank1]**2 + ddiff_pahse[:, rank1:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)
