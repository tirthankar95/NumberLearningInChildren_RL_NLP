from abc import ABC, abstractmethod
import torch.nn as nn

class NN_Strategy(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
    @abstractmethod
    def pre_process(self, state):
        pass 
    @abstractmethod
    def forward(self, state):
        pass 