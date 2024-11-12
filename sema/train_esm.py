import math
import torch
import torch.nn as nn
from transformers import Trainer

class ESM1vForTokenClassification(nn.Module):
    def __init__(self, num_labels = 2, pretrained_no = 1):
        super().__init__()
        self.num_labels = num_labels
        self.model_name = esm.pretrained.esm2_t36_3B_UR50D()  
        
        #load_model_and_alphabet_hub(self.model_name)        
        self.esm1v, self.esm1v_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.classifier = nn.Linear(1280*2, self.num_labels)

    def forward(self, token_ids, labels = None):
        outputs = self.esm1v.forward(token_ids, repr_layers=[36])['representations'][36]
        outputs = outputs[:,1:-1,:]
        logits = self.classifier(outputs)

        return SequenceClassifierOutput(logits=logits)



# loss function  
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, inputs, target, mask):    
        diff2 = (torch.flatten(inputs[:,:,1]) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) / torch.sum(mask)
        if torch.sum(mask)==0:
            return torch.sum(diff2)
        else:
            #print('loss:', result)
            return result

# train 
class MaskedRegressTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        labels = labels.squeeze().detach().cpu().numpy().tolist()
        labels = [math.log(t+1) if t!=-100 else -100 for t in labels]
        labels = torch.unsqueeze(torch.FloatTensor(labels), 0)#.cuda()
        masks = ~torch.eq(labels, -100)#.cuda()
        
        #masks = inputs.pop("masks")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = MaskedMSELoss()
        
        loss = loss_fn(logits, labels, masks)
        
        return (loss, outputs) if return_outputs else loss

