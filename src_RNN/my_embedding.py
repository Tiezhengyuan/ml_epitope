'''
text representation for RNN traing
'''
from collections import Counter, OrderedDict
import re
# torchtext=0.18.0
from torchtext.vocab import vocab
from torch.utils.data.dataset import random_split

class MyEmbedding:
    def __init__(self, data:list):
        '''
        args: data: list type. element is tuple (<label>, <text>)
        '''
        self.data = data
        self.train_ds, self.valid_ds, self.test_ds = None, None, None
    
    
    def split(self):
        '''
        split data into train:validate:test=.6:.2:.2
        '''
        num_train = int(len(self.data)*.6)
        num_valid = int(len(self.data)*.2)
        num_test = len(self.data) - num_train - num_valid
        self.train_ds, self.valid_ds, self.test_ds = random_split(
            self.data,
            [num_train, num_valid, num_test]
        )
        print('example element of data: ', self.train_ds[0])
        print('split data: ', len(self.train_ds), len(self.valid_ds), len(self.test_ds))
        return self.train_ds, self.valid_ds, self.test_ds

    def tokenizer(self, text:str, k:int):
        if k == 1:
            return list(text)
        res = []
        for i in range(0, len(text), k):
            res.append(text[i:i+k])
        return res

    def tokenize(self, k:int=1):
        '''
        need self.train_ds only
        '''
        print("\n## Step 2 tokenization: unique tokens (words)...")
        # count tokens
        input_tokens, label_tokens = Counter(), Counter()
        for label, line in self.train_ds:
            tokens = self.tokenizer(line, k)
            # words in list type
            input_tokens.update(tokens)
            label_tokens.update([label,])

        print('A sentence converted to tokens:', line, tokens)
        print('Vocab-size of input:', len(input_tokens))
        print('Vocab-size of labels:', len(label_tokens))
    
        # sort token couts of input
        sorted_by_freq_tuples = sorted(input_tokens.items(), key=lambda x: x[1], reverse=True)
        self.input_ordered_dict = OrderedDict(sorted_by_freq_tuples)
        print(self.input_ordered_dict)
        
        # sort token couts of labels
        sorted_by_freq_tuples = sorted(label_tokens.items(), key=lambda x: x[1], reverse=True)
        self.label_ordered_dict = OrderedDict(sorted_by_freq_tuples)
        counts = list(self.label_ordered_dict.values())
        print('counts of input:', counts)
        
        return self.input_ordered_dict, self.label_ordered_dict
    
    
    def build_vocab(self):
        '''
        need self.input_ordered_dict
        '''
        print("\n## Step 3 encoding: encoding each unique token into integers...")
        # Convert count value to index value (ranking)
        self.input_vocab = vocab(
            self.input_ordered_dict,
            specials=['<unk>', '<pad>', '<bos>', '<eos>'],
            special_first=True,
        )
        # self.input_vocab.insert_token("<pad>", 0)
        # self.input_vocab.insert_token("<unk>", 1)
        # default token is "<unk>"
        self.input_vocab.set_default_index(self.input_vocab['<unk>'])
        
        # labels
        self.label_vocab = vocab(
            self.label_ordered_dict,
            specials=['<unk>', '<pad>', '<bos>', '<eos>'],
            special_first=True,
        )
        # print(label_ordered_dict)
        # self.label_vocab.insert_token("<pad>", 0)
        # self.label_vocab.insert_token("<unk>", 1)
        # default token is "<unk>"
        # self.label_vocab.set_default_index(1)
        self.label_vocab.set_default_index(self.input_vocab['<unk>'])

        return self.input_vocab, self.label_vocab