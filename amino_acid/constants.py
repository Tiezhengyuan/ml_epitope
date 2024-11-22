'''
https://www.sigmaaldrich.com/US/en/technical-documents/technical-article/protein-biology/protein-structural-analysis/amino-acid-reference-chart#hydrophobicity

ApH 2 values: Normalized from Sereda et al., J. Chrom. 676: 139-153 (1994).
BpH 7 values: Monera et al., J. Protein Sci. 1: 319-329 (1995).

https://pmc.ncbi.nlm.nih.gov/articles/PMC5037676/
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



PROPERTY = {
    'A': {
        'name': 'Alanine',
        'abbreviation': 'Ala',
        'hydrophobicity': 0.61,
        'polarity': 8.1,
        'polarizability': 0.046,
        'van_der_Waals_volume': 1.00
    },
    'R': {
        'name': 'Arginine',
        'abbreviation': 'Arg',
        'hydrophobicity': 0.60,
        'polarity': 10.5,
        'polarizability': 0.291,
        'van_der_Waals_volume': 6.13
    },
    'N': {
        'name': 'Asparagine',
        'abbreviation': 'Asn',
        'hydrophobicity': 0.06,
        'polarity': 11.6,
        'polarizability': 0.134,
        'van_der_Waals_volume': 2.95
    },
    'D': {
        'name': 'Aspartic Acid',
        'abbreviation': 'Asp',
        'hydrophobicity': 0.46,
        'polarity': 13.0,
        'polarizability': 0.105,
        'van_der_Waals_volume': 2.78
    },
    'C': {'name': 'Cysteine',
        'abbreviation': 'Cyc',
        'hydrophobicity': 1.07,
        'polarity': 5.5,
        'polarizability': 0.128,
        'van_der_Waals_volume': 2.43
    },
    'Q': {
        'name': 'Glutamine',
        'abbreviation': 'Gln',
        'hydrophobicity': 0.0,
        'polarity': 10.5,
        'polarizability': 0.180,
        'van_der_Waals_volume': 3.95
    },
    'E': {
        'name': 'Glutamic Acid',
        'abbreviation': 'Glu',
        'hydrophobicity': 0.47,
        'polarity': 12.3,
        'polarizability': 0.151,
        'van_der_Waals_volume': 3.78
    },
    'G': {
        'name': 'Glycine',
        'abbreviation': 'Gly',
        'hydrophobicity': 0.07,
        'polarity': 9.0,
        'polarizability': 0.000,
        'van_der_Waals_volume': 0.00
    },
    'H': {
        'name': 'Histidine',
        'abbreviation': 'His',
        'hydrophobicity': 0.61,
        'polarity': 10.4,
        'polarizability': 0.230,
        'van_der_Waals_volume': 4.66
    },
    'I': {
        'name': 'Isoleucine',
        'abbreviation': 'Ile',
        'hydrophobicity': 2.22,
        'polarity': 5.2,
        'polarizability': 0.186,
        'van_der_Waals_volume': 4.00
    },
    'L': {
        'name': 'Leucine',
        'abbreviation': 'Leu',
        'hydrophobicity': 1.53,
        'polarity': 4.9,
        'polarizability': 0.186,
        'van_der_Waals_volume': 4.00
    },
    'K': {
        'name': 'Lysine',
        'abbreviation': 'Lys',
        'hydrophobicity': 1.15,
        'polarity': 11.3,
        'polarizability': 0.219,
        'van_der_Waals_volume': 4.77
    },
    'M': {
        'name': 'Methionine',
        'abbreviation': 'Met',
        'hydrophobicity': 1.18,
        'polarity': 5.7,
        'polarizability': 0.221,
        'van_der_Waals_volume': 4.43
    },
    'F': {
        'name': 'Phenylalanine',
        'abbreviation': 'Phe',
        'hydrophobicity': 2.02,
        'polarity': 5.2,
        'polarizability': 0.290,
        'van_der_Waals_volume': 5.89
    },
    'P': {
        'name': 'Proline',
        'abbreviation': 'Pro',
        'hydrophobicity': 1.95,
        'polarity': 8.0,
        'polarizability': 0.131,
        'van_der_Waals_volume': 2.72
    },
    'S': {
        'name': 'Serine',
        'abbreviation': 'Ser',
        'hydrophobicity': 0.05,
        'polarity': 9.2,
        'polarizability': 0.062,
        'van_der_Waals_volume': 1.60
    },
    'T': {
        'name': 'Threonine',
        'abbreviation': 'Thr',
        'hydrophobicity': 0.05,
        'polarity': 8.6,
        'polarizability': 0.108,
        'van_der_Waals_volume': 2.60
    },
    'W': {
        'name': 'Tryptophan',
        'abbreviation': 'Trp',
        'hydrophobicity': 2.65,
        'polarity': 5.4,
        'polarizability': 0.409,
        'van_der_Waals_volume': 8.08
    },
    'Y': {
        'name': 'Tyrosine',
        'abbreviation': Tyr,
        'hydrophobicity': 1.88,
        'polarity': 6.2,
        'polarizability': 0.298,
        'van_der_Waals_volume': 6.47
    },
    'V': {
        'name': 'Valine',
        'abbreviation': 'Val',
        'hydrophobicity': 1.32,
        'polarity': 5.9,
        'polarizability': 0.140,
        'van_der_Waals_volume': 3.00
    },
}


