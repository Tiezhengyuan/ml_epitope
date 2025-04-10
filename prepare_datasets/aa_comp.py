'''
amino acid composition
'''
import os
import numpy as np
import pandas as pd
import json
from collections import Counter

from constants import AA
from utils import Utils
from isolate_aa import IsolateAA
from encode_aa import EncodeAA

class AAComp:

    @staticmethod
    def isolate_epitope_seq(data_iter):
        aa_data = {}
        num_pro, num_epi = 0, 0
        for acc, rec in data_iter:
            if 'pro_seq' in rec:
                pro_seq = rec['pro_seq']
                c = IsolateAA(rec)
                aa_data[acc] = {
                    'epitope_seq': c.get_epi_seq(),
                    'other_seq': c.get_other_seq(),
                }
                num_pro += 1
                num_epi += len(rec['epitopes'])
        print(f"Number of proteins = {num_pro}")
        print(f"Number of epitopes = {num_epi}")
        return aa_data, num_pro, num_epi

    @staticmethod
    def retrieve_epitope_2aa(data_iter):
        '''
        epi_counts: count 2-AA in epitopes,
        total_epi: number of epitopes,
        other_counts: count 2-AA of non-epitopes, 
        total_other: number of non-epitopes
        '''
        total_epi, total_other = 0, 0
        epi_counts, other_counts = Counter(), Counter()
        for acc, rec in data_iter:
            c = IsolateAA(rec)
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
        for acc, rec in data_iter:
            c = IsolateAA(rec)
            aa_counts = c.kmer_counts(k)
            num_epi += c.num_epitopes()
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


################################
    @staticmethod
    def run(infile):
        outdir = os.path.dirname(infile)
        prefix = os.path.splitext(os.path.basename(infile))[0]

        # read df: default two columns
        df = pd.read_csv(infile, sep='\t', header=None, index_col=None)
        print(df.shape)

        # physcial chemical properties
        df1 = AAComp.cal_phyche(outdir, df, prefix)
        print('phy-chem:', df1.shape)

        # frequency of AA
        df2 = AAComp.cal_freq(outdir, df, prefix)
        df2 = df2.iloc[:, 2:]
        print('freq AA:', df2.shape)

        # if AA existing
        df3 = AAComp.has_aa(outdir, df, prefix)
        df3 = df3.iloc[:, 2:]
        print('if AA: ', df3.shape)

        # concatenate
        df = pd.concat([df1, df2, df3], axis=1)
        print('combined', df.shape)
        outfile = os.path.join(outdir, f'{prefix}_combined_features.txt')
        df.to_csv(outfile, header=True, index=False, sep='\t')
        print("Combine the above df and export to ", outfile)

    @staticmethod
    def cal_phyche(outdir, df, prefix):
        '''
        physcial chemical properties and export to csv
        '''
        encoder = EncodeAA()
        dfv = df.apply(
            lambda x: encoder.physical_chemical(x[0], x[1]),
            axis=1,
            result_type='expand'
        )
        dfv = dfv.fillna(0)
        # 15 features
        print('shape:', dfv.shape)
        # export
        outfile = os.path.join(outdir, f'{prefix}_physical_chemical.txt')
        dfv.to_csv(outfile, header=True, index=False, sep='\t')
        print(outfile)
        return dfv

    @staticmethod
    def cal_freq(outdir, df, prefix):
        '''
        frequency of amino acids and export to csv
        '''
        encoder = EncodeAA()
        col = ['seq', 'label'] + [f"freq_{i}" for i in encoder.aa] \
            + [f"freq_{i}" for i in encoder.aa2]

        outfile = os.path.join(outdir, f'{prefix}_frequency_aa.txt')
        with open(outfile, 'w') as f:
            f.write('\t'.join(col) + '\n')
            for row in df.itertuples():
                # print(row)
                s = encoder.frequency_aa(row._1, row._2)
                f.write('\t'.join(s) + '\n')
        print(outfile)
        dfv = pd.read_csv(outfile, sep='\t', header=0, index_col=None)
        return dfv

    @staticmethod
    def has_aa(outdir, df, prefix):
        '''
        detect if AA is existing export to csv
        '''
        encoder = EncodeAA()
        col = ['seq', 'label'] + [f"has_{i}" for i in encoder.aa] \
            + [f"has_{i}" for i in encoder.aa2]

        outfile = os.path.join(outdir, f'{prefix}_aa_existing.txt')
        with open(outfile, 'w') as f:
            f.write('\t'.join(col) + '\n')
            for row in df.itertuples():
                # print(row)
                s = encoder.existing(row._1, row._2)
                f.write('\t'.join(s) + '\n')
        print(outfile)
        dfv = pd.read_csv(outfile, sep='\t', header=0, index_col=None)
        return dfv

    @staticmethod
    def represent_seq_1(infile):
        outdir = os.path.dirname(infile)
        prefix = os.path.splitext(os.path.basename(infile))[0]

        # read df: default two columns
        df = pd.read_csv(infile, sep='\t', header=None, index_col=None)
        df.columns = ['seq', 'label']
        print(df.shape)

        encoder = EncodeAA()
        phy_che = df['seq'].map(lambda x: '. '.join(encoder.physcial_chemical(x)))
        dfv= pd.DataFrame({
            'text': phy_che,
            'label': df['label'],
        })
        print(dfv)

        # export
        outfile = os.path.join(outdir, f'{prefix}_physical_chemical_seq.csv')
        dfv.to_csv(outfile, header=True, index=False)
        print(outfile)

    @staticmethod
    def represent_seq_2(infile):
        outdir = os.path.dirname(infile)
        prefix = os.path.splitext(os.path.basename(infile))[0]

        # read df: default two columns
        df = pd.read_csv(infile, sep='\t', header=None, index_col=None)
        df.columns = ['seq', 'label']
        print(df.shape)

        encoder = EncodeAA()
        # phy_che = df['seq'].map(encoder.physical_chemical_text)
        smiles = df['seq'].map(encoder.aa_smiles)
        dfv= pd.DataFrame({
            'seq': df['seq'],
            'text': smiles,
            'label': df['label'],
        })
        dfv=dfv.dropna()
        dfv=dfv[dfv['text'].str.len()>=6]
        print(dfv.shape)
        print(dfv)

        # export
        outfile = os.path.join(outdir, f'{prefix}_physical_chemical_seq.csv')
        dfv.to_csv(outfile, header=True, index=False)
        print(outfile)
