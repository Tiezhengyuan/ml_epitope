import torch
import torch.nn as nn

class MyCollate:

    def __init__(self, text_pipeline, label_pipeline, feature_pipeline=None):
        self.text_pipeline = text_pipeline
        self.label_pipeline = label_pipeline
        self.feature_pipeline = feature_pipeline
    
    def __call__(self, batch) -> dict:
        '''
        return: tokens of texts, labels, lengths, features (other features)
        '''
        # wrap the encode and transformation function
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # process data
        label_list, text_list, lengths, features = [], [], [], []
        for _label, _text in batch:
            # print(_label, _text)
            # labels
            label_list.append(self.label_pipeline(_label))
            # texts
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            # lengths of texts
            lengths.append(processed_text.size(0))
            #optional: other features
            if self.feature_pipeline:
                feature = torch.tensor(self.feature_pipeline(_text), dtype=torch.float32)
                features.append(feature)

        # padding is appended to the end of token vector of each sentence
        padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
        label_list = torch.tensor(label_list, dtype=torch.float32)
        lengths = torch.tensor(lengths)
        input_batch = {
            'texts': padded_text_list.to(device),
            'labels': label_list.to(device),
            'lengths': lengths.to(device),
        }
        # optional: other features
        if features:
            features = torch.stack(features)
            input_batch['features'] = features.to(device)
        return input_batch

    # slice sentence by word.
    def tokenizer_single(self, text:str):
        return list(text)
    
    def tokenizer_kmer(self, text:str):
        step, res = 6, []
        for i in range(0, len(text)-step+1):
            res.append(text[i:i+step])
        return res