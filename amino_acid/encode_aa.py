import numpy as np
from collections import Counter
from constants import PROPERTY

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

    def freq_1aa(self, seq:str):
        counts = Counter(seq)
        n = len(seq)
        freq = [counts[i]/n for i in self.aa]
        return freq


#################
    def mean_comp(self, seq:str, label:int=None) -> np.array:
        '''
        mean properties given a epitope sequence
        '''
        res = [
            np.mean(self.hydrophobicity_ph7(seq)),
            np.mean(self.hydrophobicity(seq)),
            np.mean(self.polarity(seq)),
            np.mean(self.polarizability(seq)),
            np.mean(self.van_der_Waals_volume(seq)),
        ]
        res += self.freq_1aa(seq)
        res = np.round(res, decimals=3)
        if label is not None:
            res = np.append(res, label)

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