from typing import NamedTuple
import numpy as np


class AxInfo(NamedTuple):
    n : int = 1
    o: float = 0.
    d: float = 1.
    l: str = "undefined"
    
    def to_string(self, ax: int = 1):
        return "n%s=%s o%s=%s d%s=%s label%s='%s'\n" % (ax, self.n, ax, self.o, ax, self.d, ax, self.l)
    
    def plot(self):
        return np.arange(self.n) * self.d + self.o
    
    @property
    def last(self):
        return self.o + (self.n-1) * self.d
