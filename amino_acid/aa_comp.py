'''
amino acid composition
'''
import os
import numpy as np
import pandas as pd
import json
from collections import Counter

from utils import Utils
from isolate_aa import IsolateAA
from constants import AA

class AAComp:

    @staticmethod
    def isolate_epitope_seq(data_iter):

        aa_data = {}
        num_pro, num_epi = 0, 0
        for data, path in data_iter:
            print(path)
            for acc in data:
                pro_seq = data[acc]['pro_seq']
                c = IsolateAA(data[acc])
                aa_data[acc] = {
                    'epitope_seq': c.get_epi_seq(),
                    'other_seq': c.get_other_seq(),
                }
                num_pro += 1
                num_epi += len(data[acc]['epitopes'])
                if len((aa_data[acc]['other_seq']) == len(pro_seq):
                    print(acc)
                    print(data[acc]['epitopes'])
                    print(aa_data[acc])
        print(f"Number of proteins = {num_pro}")
        print(f"Number of epitopes = {num_epi}")
        return aa_data, num_pro, num_epi

    @staticmethod
    def retrieve_epitope_2aa(data_iter):
        total_epi, total_other = 0, 0
        epi_counts, other_counts = Counter(), Counter()
        for data, path in data_iter:
            print(path)
            for acc in data:
                c = IsolateAA(data[acc])
                epi_len = len(c.get_epi_seq())
                other_len = len(c.get_other_seq())
                epi_seq, other_seq = c.isolate_2aa()
                epi_counts += Counter(epi_seq)
                total_epi += epi_len
                other_counts += Counter(other_seq)
                total_other += other_len
        return epi_counts, total_epi, other_counts, total_other

    @staticmethod
    def retrieve_frequency(data_iter, k:int):
        num_epi = 0
        epi_counts = {}
        for data, path in data_iter:
            for acc in data:
                aa_counts, n = IsolateAA(data[acc]).slice_aa(k)
                num_epi += n
                for aa, count in aa_counts.items():
                    if aa not in epi_counts:
                        epi_counts[aa] = count
                    else:
                        epi_counts[aa] += count
        aa_freq = pd.DataFrame({
            'aa': epi_counts.keys(),
            'counts': epi_counts.values(),
            'freq': [i*100/num_epi for i in epi_counts.values()],
            'hydrophobicity': AAComp.cal_hydro(epi_counts.keys())
        }).sort_values('freq', ascending=False).dropna()
        return aa_freq, num_epi
    
    @staticmethod
    def cal_hydro(sliced_aa:list):
        hydrophobicity = []
        for aa_seg in sliced_aa:
            hydro = 0
            for aa in list(aa_seg):
                if aa in AA:
                    hydro += AA[aa]['hydrophobicity']
                else:
                    hydro = None
                    break
            hydrophobicity.append(hydro)
        return hydrophobicity
            