import torch
import torch.nn as nn

import esm
from transformers.modeling_outputs import SequenceClassifierOutput


class ESM2t30(nn.Module):

    def __init__(self, num_labels = 2, pretrained_no = 1):
        super().__init__()
        self.num_labels = num_labels
        #load_model_and_alphabet_hub(self.model_name)        
        self.esm, self.esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.classifier = nn.Linear(640, self.num_labels)

    def forward(self, token_ids, labels = None):
        # num_layers = 33
        outputs = self.esm.forward(token_ids, repr_layers=[30])['representations'][30]
        outputs = outputs[:,1:-1,:]
        logits = self.classifier(outputs)
        return SequenceClassifierOutput(logits=logits)


class ESM2t33(nn.Module):

    def __init__(self, num_labels = 2, pretrained_no = 1):
        super().__init__()
        self.num_labels = num_labels
        #load_model_and_alphabet_hub(self.model_name)        
        self.esm, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.classifier = nn.Linear(1280, self.num_labels)

    def forward(self, token_ids, labels = None):
        # num_layers = 33
        outputs = self.esm.forward(token_ids, repr_layers=[33])['representations'][33]
        outputs = outputs[:,1:-1,:]
        logits = self.classifier(outputs)
        return SequenceClassifierOutput(logits=logits)



class ESM2t36(nn.Module):

    def __init__(self, num_labels = 2, pretrained_no = 1):
        super().__init__()
        self.num_labels = num_labels
        #load_model_and_alphabet_hub(self.model_name)        
        self.esm, self.esm1v_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.classifier = nn.Linear(2560, self.num_labels)

    def forward(self, token_ids, labels = None):
        # num_layers = 36
        outputs = self.esm.forward(token_ids, repr_layers=[36])['representations'][36]
        outputs = outputs[:,1:-1,:]
        logits = self.classifier(outputs)
        return SequenceClassifierOutput(logits=logits)
        