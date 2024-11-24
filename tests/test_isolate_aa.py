from unittest import TestCase
from ddt import ddt, data, unpack

from amino_acid.isolate_aa import IsolateAA


example = {
    'epitopes': {
    "1243835": {"end": 37, "start": 20, "id": 1243835,
        "seq": "TTTVQYNPSEQYQPYPEQ"},
    "1243837": {"end": 46, "start": 28, "id": 1243837,
        "seq": "SEQYQPYPEQQEPFVQQQQ"},
    "1243838": {"end": 55, "start": 37, "id": 1243838,
        "seq": "QQEPFVQQQQPFVQQQQPF"},
    "1243843": {"end": 42, "start": 31, "id": 1243843,
        "seq": "YQPYPEQQEPFV"},
    "1243845": {"end": 41, "start": 20, "id": 1243845,
        "seq": "TTTVQYNPSEQYQPYPEQQEPF"}
    },
    'pro_seq': "MKTFLIIALLAMAVATATATTTVQYNPSEQYQPYPEQQEPFVQQQQPFVQQQQPF" + \
       "VQQQQMFLQPLLQQQLNPCKQFLVQQCSPVAAVPFLRSQILRQAICQVTRQQCCRQLAQIP" + \
        "EQLRCPAIHSVVQSIILQQQQQQQQFIQPQLQQQVFQPQLQLQQQVFQPQLQQQVFQPQLQ" + \
        "QVFNQPQMQGQIEGMRAFALQALPAMCDVYVPPQCPVATAPLGGF"
}

@ddt
class TestIsolateAA(TestCase):

    def test_example(self):
        c = IsolateAA(example)
        # initialize
        assert c.pro_len == 222
        assert ''.join(c.pro_aa) == example['pro_seq']

        # epitope sequence
        epi_seq = c.get_epi_seq()
        assert epi_seq == 'TTTVQYNPSEQYQPYPEQQEPFVQQQQPFVQQQQPF'
        num_epi = c.num_epitopes()
        assert num_epi == 5
        # 
        epi_seq, other_seq = c.isolate_2aa()
        assert epi_seq[:5] == ['TT', 'TT', 'TV', 'VQ', 'QY',]
        assert other_seq[:5] == ['MK', 'KT', 'TF', 'FL', 'LI',]
    

    @data(
        [9, {
            'YNPSEQYQP': 2, 'NPSEQYQPY': 2, 'TVQYNPSEQ': 2, 
            'PSEQYQPYP': 2, 'TTVQYNPSE': 2, 'EQYQPYPEQ': 3, 
            'SEQYQPYPE': 3, 'VQYNPSEQY': 2, 'QYNPSEQYQ': 2, 
            'TTTVQYNPS': 2, 'QYQPYPEQQ': 2, 'PEQQEPFVQ': 1, 
            'YPEQQEPFV': 2, 'PYPEQQEPF': 3, 'YQPYPEQQE': 3, 
            'EQQEPFVQQ': 1, 'QQEPFVQQQ': 2, 'QPYPEQQEP': 3, 
            'QEPFVQQQQ': 2, 'QQPFVQQQQ': 1, 'QPFVQQQQP': 1, 
            'FVQQQQPFV': 1, 'QQQQPFVQQ': 1, 'QQQPFVQQQ': 1, 
            'EPFVQQQQP': 1, 'VQQQQPFVQ': 1, 'PFVQQQQPF': 1
        }],
        [20, {
            'TVQYNPSEQYQPYPEQQEPF': 1,
            'TTTVQYNPSEQYQPYPEQQE': 1, 
            'TTVQYNPSEQYQPYPEQQEP': 1
        }],
    )
    @unpack
    def test_kmer_counts(self, size, expect_counts):
        # slice epitopes
        c = IsolateAA(example)
        epi_counts = c.kmer_counts(size)
        assert epi_counts == expect_counts

    @data(
        [9, {'PEQQEPFVQ', 'TVQYNPSEQ', 'QQQPFVQQQ', 'YQPYPEQQE', 
            'QPYPEQQEP', 'QPFVQQQQP', 'EQYQPYPEQ', 'QYQPYPEQQ', 
            'VQQQQPFVQ', 'YPEQQEPFV', 'QQPFVQQQQ', 'TTTVQYNPS', 
            'EPFVQQQQP', 'SEQYQPYPE', 'QQEPFVQQQ', 'EQQEPFVQQ', 
            'FVQQQQPFV', 'QYNPSEQYQ', 'QEPFVQQQQ', 'PYPEQQEPF', 
            'TTVQYNPSE', 'PFVQQQQPF', 'QQQQPFVQQ', 'NPSEQYQPY', 
            'PSEQYQPYP', 'VQYNPSEQY', 'YNPSEQYQP'}],
        [20, {'ATTTVQYNPSEQYQPYPEQQ', 'PSEQYQPYPEQQEPFVQQQQ', 
            'QQEPFVQQQQPFVQQQQPFV', 'SEQYQPYPEQQEPFVQQQQP', 
            'TTVQYNPSEQYQPYPEQQEP', 'TVQYNPSEQYQPYPEQQEPF', 
            'TTTVQYNPSEQYQPYPEQQE'}],
    )
    @unpack
    def test_kmer_expand(self, size, expect):
        # slice epitopes
        c = IsolateAA(example)
        segment_seq = c.slice_kmer_expand(size)
        assert set(segment_seq) == expect

    @data(
        [9, {'VQQQQPFVQ', 'PYPEQQEPF', 'QYNPSEQYQ', 'NPSEQYQPY', 'QPYPEQQEP'}],
        [20, {'TTVQYNPSEQYQPYPEQQEP', 'PSEQYQPYPEQQEPFVQQQQ', 
            'SEQYQPYPEQQEPFVQQQQP', 'QQEPFVQQQQPFVQQQQPFV', 
            'ATTTVQYNPSEQYQPYPEQQ'}],
    )
    @unpack
    def test_shrink_expand(self, size, expect):
        # slice epitopes
        c = IsolateAA(example)
        segment_seq = c.slice_shrink_expand(size)
        assert set(segment_seq) == expect
        assert len(segment_seq) == c.num_epitopes()
    
    @data(
        [9, 1],
        [9, 10],
        [30, 10],
        # random seq is longer than the target seq
        [10000, 2],
    )
    @unpack
    def test_random_other_seq(self, size, num):
        c = IsolateAA(example)
        segment_seq = c.random_other_seq(size, num)
        if segment_seq:
            assert len(segment_seq) == num
            other_seq = c.get_other_seq()
            res = set([i in other_seq for i in segment_seq])
            assert res == {True,}
        else:
            assert segment_seq == []