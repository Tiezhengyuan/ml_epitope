'''
https://www.sigmaaldrich.com/US/en/technical-documents/technical-article/protein-biology/protein-structural-analysis/amino-acid-reference-chart#hydrophobicity
'''
# hydrophobicity index at pH7
AA = {
    "A":{'abb': 'ala', 'hydrophobicity': 41,},
    "R":{'abb': 'arg', 'hydrophobicity': -14,},
    "N":{'abb': 'asn', 'hydrophobicity': -28,},
    "D":{'abb': 'asp', 'hydrophobicity': -55,},
    "C":{'abb': 'cys', 'hydrophobicity': 49,},
    "E":{'abb': 'glu', 'hydrophobicity': -31,},
    "Q":{'abb': 'gln', 'hydrophobicity': -10,},
    "G":{'abb': 'gly', 'hydrophobicity': 0,},
    "H":{'abb': 'his', 'hydrophobicity': 8,},
    "I":{'abb': 'ile', 'hydrophobicity': 99,},
    "L":{'abb': 'leu', 'hydrophobicity': 97,},
    "K":{'abb': 'lys', 'hydrophobicity': -23,},
    "M":{'abb': 'met', 'hydrophobicity': 74,},
    "F":{'abb': 'phe', 'hydrophobicity': 100,},
    "P":{'abb': 'pro', 'hydrophobicity': -46,}, #PH2
    "S":{'abb': 'ser', 'hydrophobicity': -5,},
    "T":{'abb': 'thr', 'hydrophobicity': 13,},
    "W":{'abb': 'trp', 'hydrophobicity': 97,},
    "Y":{'abb': 'tyr', 'hydrophobicity': 63,},
    "V":{'abb': 'val', 'hydrophobicity': 76,},
}
