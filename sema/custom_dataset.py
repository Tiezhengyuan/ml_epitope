"""
A class to represent a sutable data set for model. 

convert original pandas data frame to model set,
where 'token ids' is ESM-1v embedings corresponed to protein sequence (max length 1022 AA)
and 'lables' is a contact number values
"""
import esm
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PdbDataset(Dataset):

    def __init__(self, converter, df):
        """
        batch_converter (function): ESM function callable to convert an unprocessed
            (labels + strings) batch to a processed (labels + tensor) batch.
        df (pandas.DataFrame): dataframe with two columns: 
                0 -- protein sequence in string ('GLVM') or list (['G', 'L', 'V', 'M']) format
                1 -- contcat number values in list [0, 0.123, 0.23, -100, 1.34] format
        # label_type (str): type of model: regression or binary
        """
        self.batch_converter = converter
        self.df = df
        # self.label_type = label_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        '''
        '''
        item = {}
        # residual_aa
        aa = [('' , ''.join(self.df.iloc[idx,0])[:1022])]
        _, _, esm1b_batch_tokens = self.batch_converter(aa)
        item['token_ids'] = esm1b_batch_tokens
        # contact_number_binary
        contact = torch.tensor(self.df.iloc[idx, 1][:1022], dtype=torch.float16)
        item['labels'] = torch.unsqueeze(contact,0).to(torch.float64)
        return item

