from typing import NamedTuple

import numpy as np


class AxInfo(NamedTuple):
    """
    Store information about vectors' axis

    Attributes:
        N: number of samples along the axis
        o: axis origin value (i.e., value the first sample)
        d: sampling/discretization step
        l: label of the axis
        last: value of the last sample
    """
    n: int = 1
    o: float = 0.
    d: float = 1.
    l: str = "undefined"
    
    def to_string(self, ax: int = 1):
        """
        Create a description of the axis

        Args:
            ax: axis number for printing (for SEPlib compatibility)

        Returns: string

        """
        return "n%s=%s o%s=%s d%s=%s label%s='%s'\n" % (ax, self.n, ax, self.o, ax, self.d, ax, self.l)
    
    def plot(self):
        """
        Create a np.ndarray of the axis, useful for plotting

        """
        return np.arange(self.n) * self.d + self.o
    
    @property
    def last(self):
        return self.o + (self.n - 1) * self.d
