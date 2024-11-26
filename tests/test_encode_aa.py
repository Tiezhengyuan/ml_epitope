import numpy as np
from unittest import TestCase
from ddt import ddt, data, unpack

from amino_acid.encode_aa import EncodeAA


@ddt
class TestEncodeAA(TestCase):
    def setUp(self):
        self.c = EncodeAA()

    @data(
        ['PEQQEPFVQ', ],
        ['XXX', ],
        # ['GKTVPLPPSSAM',],
        ['PEYT',],
    )
    @unpack
    def test_(self, seq):
        # vector
        res = self.c.vector_1d(seq)
        # assert len(res) == len(seq) * 5
        # assert res.dtype == np.float16
        print(res, res.dtype)
        
        res = self.c.vector_2d(seq)
        # assert res.shape == (len(seq), 5)
        # assert res.dtype == np.float16

