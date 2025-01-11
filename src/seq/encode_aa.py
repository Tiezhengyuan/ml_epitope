import numpy as np
import pandas as pd
from collections import Counter

from .constants import PROPERTY

class EncodeAA:
    def __init__(self):
        self.hydro_ph7 = {s:PROPERTY[s]['hydrophobicity_ph7'] for s in PROPERTY}
        self.hydro = {s:PROPERTY[s]['hydrophobicity'] for s in PROPERTY}
        self.polar = {s:PROPERTY[s]['polarity'] for s in PROPERTY}
        self.pz = {s:PROPERTY[s]['polarizability'] for s in PROPERTY}
        self.vdw = {s:PROPERTY[s]['van_der_Waals_volume'] for s in PROPERTY}
        self.aa = list(PROPERTY)

    def hydrophobicity_ph7(self, seq:str):
        res = [self.hydro_ph7.get(s, 0) for s in seq]
        return res

    def hydrophobicity(self, seq:str):
        res = [self.hydro.get(s, 0) for s in seq]
        return res

    def polarity(self, seq:str):
        res = [self.polar.get(s, 0) for s in seq]
        return res

    def polarizability(self, seq:str):
        res = [self.pz.get(s, 0) for s in seq]
        return res

    def van_der_Waals_volume(self, seq:str):
        res = [self.vdw.get(s, 0) for s in seq]
        return res

    def freq_1aa(self, seq:str) -> pd.Series:
        '''
        frequency of 2-AA
        '''
        counts = Counter(seq)
        n = len(seq)
        freq = [counts[i]/n for i in self.aa]
        freq = pd.Series(freq, index=[f"freq_{i}" for i in self.aa])
        return freq
    
    def freq_2aa(self, seq:str) -> pd.Series:
        '''
        frequency of 2-AA
        '''
        freq = {}
        for i in range(0, len(seq)-1):
            aa = seq[i:i+2]
            key = f"freq_{aa}"
            if key not in freq:
                freq[key] = 1
            else:
                freq[key] += 1
        freq = pd.Series(freq)/len(seq)
        return freq



#################
    def physical_chemical(self, seq:str, label:int=None) -> pd.Series:
        '''
        physical chemical properties given a epitope sequence
        '''
        # physical-chemical properties
        names = ['hydro_ph7', 'hydro', 'polar', 'polar_stab', 'VDWV']
        phyche = [
            self.hydrophobicity_ph7(seq),
            self.hydrophobicity(seq),
            self.polarity(seq),
            self.polarizability(seq),
            self.van_der_Waals_volume(seq),
        ]
        phyche_mean = pd.Series([np.mean(i) for i in phyche],
            index=[f"mean_{i}" for i in names])
        phyche_median = pd.Series([np.median(i) for i in phyche],
            index=[f"median_{i}" for i in names])
        phyche_var = pd.Series([np.var(i) for i in phyche],
            index=[f"variance_{i}" for i in names])

        # outcome
        res = pd.concat([phyche_mean, phyche_median, phyche_var])
        res = res.round(decimals=3)
        if label is not None:
            res['label'] = label
        return res

    def frequency_aa(self, seq:str, label:int=None) -> pd.Series:
        '''
        frequency of amino acids
        '''
        freq1a = self.freq_1aa(seq)
        freq2a = self.freq_2aa(seq)

        # outcome
        res = pd.concat([freq1a, freq2a])
        res = res.round(decimals=3)
        if label is not None:
            res['label'] = label
        return res

    def vector_1d(self, seq:str, label:int=None) -> np.array:
        res = self.hydrophobicity_ph7(seq) + self.hydrophobicity(seq) + \
                self.polarity(seq) + self.polarizability(seq) + \
                self.van_der_Waals_volume(seq)
        if label is not None:
            res = np.append(res, label)
        return res
    
    def vector_2d(self, seq:str) -> np.array:
        res = self.vector_1d(seq)
        m = len(seq)
        n = int(len(res)/m)
        return res.reshape(m, n)