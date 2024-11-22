import os
import numpy as np
import json
from collections import Counter

from utils import Utils


class IsolateAA:

    def __init__(self, record:dict):
        self.pro_seq = record['pro_seq']
        self.epitopes = record['epitopes']
        self.pro_aa = None
        self.is_epi = None

    def filter_epitopes(self):
        '''
        retrieve epitope sequence and non-epitope seq
        '''
        self.pro_aa = np.array(list(self.pro_seq))
        self.is_epi = np.zeros(len(self.pro_seq))
        for item in self.epitopes.values():
            start = int(item['start']) - 1
            end = int(item['end'])
            self.is_epi[start:end] = 1
        return self.pro_aa, self.is_epi

    def isolate_1aa(self):
        '''
        retrieve epitope sequence and non-epitope seq
        '''
        if not self.is_epi:
            self.filter_epitopes()
        epi_seq = self.pro_aa[self.is_epi==1]
        other_seq = self.pro_aa[self.is_epi==0]
        return epi_seq, other_seq

    def isolate_2aa(self):
        '''
        retrieve epitope sequence and non-epitope seq
        '''
        if not self.is_epi:
            self.filter_epitopes()
        epi_seq, other_seq = [], []
        for i in range(len(self.pro_seq)-1):
            if self.is_epi[i] == self.is_epi[i+1]:
                aa2 = self.pro_seq[i:i+2]
                if self.is_epi[i] == 0:
                    other_seq.append(aa2)
                else:
                    epi_seq.append(aa2)
        return epi_seq, other_seq
    
    def slice_aa(self, k:int):
        epi_counts = {}
        for item in self.epitopes.values():
            epi_aa = []
            seq = item['seq']
            for i in range(0, len(seq)-k+1):
                epi_aa.append(seq[i:i+k])
            # count frequency
            for i in list(set(epi_aa)):
                if i not in epi_counts:
                    epi_counts[i] = 1
                else:
                    epi_counts[i] += 1
        return epi_counts, len(self.epitopes)
