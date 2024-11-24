from unittest import TestCase
from ddt import ddt, data, unpack

from amino_acid.utils import Utils

@ddt
class TestUtils(TestCase):

    @data (
        [20, 3, 5, 8, (0, 8)],
        [20, 0, 4, 8, (0, 8)],
        [20, 17, 19, 8, (11, 19)],
        [20, 3, 5, 9, (0, 9)],
        [20, 0, 4, 9, (0, 9)],
        [20, 17, 19, 9, (10, 19)],
        [20, 10, 10, 5, (8, 13)],
        # uncommon, return None
        # exact size
        [20, 3, 5, 2, None],
        # loner than size
        [20, 3, 15, 8, None],
        # seq_len is short
        [5, 3, 5, 8, None],
        # size is short
        [20, 3, 10, 5, None],
        # wrong start or end
        [20, -3, 10, 5, None],
        [20, -3, -10, 5, None],
        [20, 10, 3, 5, None],
    )
    @unpack
    def test_expand(self, seq_len, start, end, size, expect):
        res = Utils.expand(seq_len, start, end, size)
        assert res == expect

    @data (
        [3, 10, 4, (4, 8)],
        [3, 10, 5, (4, 9)],
        [3, 100, 50, (26, 76)],
        # uncommon, return None
        # exact size
        [3, 5, 2, None],
        # loner than size
        [3, 5, 8, None],
        # wrong start or end
        [-3, 10, 5, None],
        [-3, -10, 5, None],
        [10, 3, 5, None],
        [3, 3, 5, None],
    )
    @unpack
    def test_shrink(self, start, end, size, expect):
        res = Utils.shrink(start, end, size)
        assert res == expect