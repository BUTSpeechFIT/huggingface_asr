# pylint: skip-file

from typing import Union, List, Optional
import os
import pickle


import torch


class Feature_Dumper(torch.nn.Module):
    """Dumping featured to file and forward

    """
    def __init__(self,
        counter: int = 0,
        output_dir: str = "",
        prefix: str = "",
        dump_end: int = 20,
        dump_start: int = 10
    ):
        super().__init__()
        self.counter = counter
        self.output_dir = output_dir
        self.prefix = prefix
        self.dump_end = dump_end
        self.dump_start = dump_start


    def forward(self, x):
        print(x)
        self.counter = self.counter + 1
        if (self.counter >= self.dump_start) and (self.counter < self.dump_end):
            with open(os.path.join(self.output_dir, f"{self.prefix}_{self.counter:010}.fea.pkl"), "wb") as f:
                pickle.dump(x, f)
        print(x.unsqueeze(0))
        return x.unsqueeze(0)

