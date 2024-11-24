import numpy as np

from amino_acid.constants import PROPERTY

class EncodeAA:
    def __init__(self):
        self.hydro_ph7 = {s:PROPERTY[s]['hydrophobicity_ph7'] for s in PROPERTY}
        self.hydro = {s:PROPERTY[s]['hydrophobicity'] for s in PROPERTY}
        self.polar = {s:PROPERTY[s]['polarity'] for s in PROPERTY}
        self.pz = {s:PROPERTY[s]['polarizability'] for s in PROPERTY}
        self.vdw = {s:PROPERTY[s]['van_der_Waals_volume'] for s in PROPERTY}
    
    def vector_1d(self, seq:str) -> np.array:
        res = np.concatenate([
            self.hydrophobicity_ph7(seq),
            self.hydrophobicity(seq),
            self.polarity(seq),
            self.polarizability(seq),
            self.van_der_Waals_volume(seq),
        ]).astype(np.float16)
        return res
    
    def vector_2d(self, seq:str) -> np.array:
        res = self.vector_1d(seq)
        m = len(seq)
        n = int(len(res)/m)
        return res.reshape(m, n)

    def hydrophobicity_ph7(self, seq:str):
        res = [self.hydro_ph7.get(s, 0) for s in seq]
        return np.array(res)

    def hydrophobicity(self, seq:str):
        res = [self.hydro.get(s, 0) for s in seq]
        return np.array(res)

    def polarity(self, seq:str):
        res = [self.polar.get(s, 0) for s in seq]
        return np.array(res)

    def polarizability(self, seq:str):
        res = [self.pz.get(s, 0) for s in seq]
        return np.array(res)

    def van_der_Waals_volume(self, seq:str):
        res = [self.vdw.get(s, 0) for s in seq]
        return np.array(res)
