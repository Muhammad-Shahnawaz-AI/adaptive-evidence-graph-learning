from abc import ABC

import torch.nn as nn

class BaseModule(

    nn.Module,

    ABC

):

    def __init__(self):

        super().__init__()

    @property

    def device(self):

        return next(self.parameters()).device