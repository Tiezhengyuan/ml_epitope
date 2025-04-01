import os
import numpy as np
import json
from collections import Counter

from utils import Utils


class IsolateAA:

    def __init__(self, record:dict):
        self.pro_seq = record.get('pro_seq', '')
        self.epitopes = record['epitopes']
        self.pro_len = len(self.pro_seq)

        # retrieve epitope sequence and non-epitope seq
        self.pro_aa = np.array(list(self.pro_seq))
        self.is_epi = np.zeros(self.pro_len)
        for item in self.epitopes:
            start = int(item['start']) - 1
            end = int(item['end'])
            self.is_epi[start:end] = 1
        # index of non-epitope
        self.ix = np.where(self.is_epi == 0)[0]

    def get_epi_seq(self):
        return ''.join(self.pro_aa[self.is_epi==1])

    def get_other_seq(self):
        return ''.join(self.pro_aa[self.is_epi==0])

    def num_epitopes(self):
        return len(self.epitopes)

    def isolate_1aa(self):
        '''
        retrieve epitope sequence and non-epitope seq
        '''
        epi_seq = self.pro_aa[self.is_epi==1]
        other_seq = self.pro_aa[self.is_epi==0]
        return epi_seq, other_seq

    def isolate_2aa(self):
        '''
        retrieve epitope sequence and non-epitope seq
        '''
        epi_seq, other_seq = [], []
        for i in range(self.pro_len - 1):
            if self.is_epi[i] == self.is_epi[i+1]:
                aa2 = self.pro_seq[i:i+2]
                if self.is_epi[i] == 0:
                    other_seq.append(aa2)
                else:
                    epi_seq.append(aa2)
        return epi_seq, other_seq
    
    def kmer_counts(self, size:int):
        epi_counts = {}
        for item in self.epitopes:
            epi_aa = []
            seq = item['seq']
            for i in range(0, len(seq) - size + 1):
                epi_aa.append(seq[i:i + size])
            # count frequency
            for i in list(set(epi_aa)):
                if i not in epi_counts:
                    epi_counts[i] = 1
                else:
                    epi_counts[i] += 1
        return epi_counts
    
    def slice_kmer_expand(self, size:int):
        '''
        arg: size is window size
        1. slice all epiptops into <size>
        2. longer k-mer slidding, 
        3. shorter. expanding to <size> from two sides
        Note: some sliced seq may be shorter
        '''
        segment_seq = set()
        for item in self.epitopes:
            start = item['start'] -1
            end = item['end']
            seq = item['seq']
            if len(seq) >= size:
                for i in range(0, len(seq)-size+1):
                    segment_seq.add(seq[i:i+size])
            else:
                _start, _end = Utils.expand(self.pro_len, start, end, size)
                segment_seq.add(self.pro_seq[_start:_end])
        return list(segment_seq)

    def slice_shrink_expand(self, size:int) -> list:
        '''
        arg: size is window size
        1. slice all epiptops into <size>
        2. longer, shrink to <size> from two sides, 
        3. shorter. expanding to <size> from two sides
        '''
        segment_seq = set()
        for item in self.epitopes:
            start = item['start'] -1
            end = item['end']
            seq = item['seq']
            if len(seq) == size:
                segment_seq.add(seq)
            elif len(seq) > size:
                _start, _end = Utils.shrink(start, end, size)
                segment_seq.add(self.pro_seq[_start:_end])
            else:
                _start, _end = Utils.expand(self.pro_len, start, end, size)
                segment_seq.add(self.pro_seq[_start:_end])
        return list(segment_seq)


    def random_other_seq(self, size:int, num:int) -> list:
        '''
        random sequence from non-epitope the same number as sliced epitopes
        Note: it is possible that no random seq could be generated
            or number is less then the required
        '''
        # get index
        m = len(self.ix)/size
        if m > num:
            if m > num * 2:
                if m > num * 4:
                    num = num * 3
                else:
                    num = num * 2
            else:
                num = int(m)
        # try 10x times
        segment_seq, n = set(), 0
        while len(segment_seq) < num and n < num * 10:
            start = np.random.choice(ix)
            end = start + size
            if end < self.pro_len and self.is_epi[end] == 0:
                other_seq = self.pro_seq[start:end]
                segment_seq.add(other_seq)
            n += 1
        return list(segment_seq)
    
    def random_size_seq(self, num:int, size:int=None):
        '''
        num: given one epitopes, get num of non-epitopes
        size: in default epitopes is as long as non-epitopes
        '''
        segment_seq = {}
        for item in self.epitopes:
            if size is None:
                size = len(item['seq'])
            n, m = 0, 0
            # try not more than 100times
            while n < num and m < 100:
                start = np.random.choice(self.ix)
                end = start + size
                # no overlapping between epitope and non-epitope
                if end < self.pro_len and sum(self.is_epi[start:end]) == 0:
                    other_seq = self.pro_seq[start:end]
                    if other_seq not in segment_seq:
                        segment_seq[other_seq] = 1
                        n += 1
                m += 1
        return list(segment_seq)