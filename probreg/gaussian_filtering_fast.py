from __future__ import print_function
from __future__ import division
import numpy as np
from . import _preload_lattice


class Permutohedral_fast(object):
    def __init__(self, N,M,d,with_blur=True):
        self._impl = _preload_lattice.Permutohedral_p(N,M,d,with_blur)

    def init_with_val(self,features,in_tmp,with_blur):
        return self._impl.init_with_val(features.T,in_tmp.T,with_blur)

    def apply(self, out, feature):
        return self._impl.apply(out.T, feature.T)
