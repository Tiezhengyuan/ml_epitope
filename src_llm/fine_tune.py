from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class FineTune:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def model_predict(model, tokenizer, dataset, batch_size=8):
        '''
        example BERT
        '''
        model.eval()

        stat = {
            'labels': [],
            'predictions': [],
            'scores': [],
        }
        # convert test dataset to dataloader
        eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in eval_dataloader:
            # true labels
            stat['labels'].extend(batch["label"].tolist())
            with torch.no_grad():
                inputs = tokenizer(
                    batch['text'], 
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                outputs = model(**inputs)
                logits = outputs.logits
                # probabilities of label=1
                scores = F.softmax(logits, dim=-1)[:,-1].tolist()
                stat['scores'].extend(scores)
                # predicted labels
                predictions = torch.argmax(logits, dim=-1)
                stat['predictions'].extend(predictions.tolist())
        return stat

    @staticmethod
    def trainer_predict(trainer, dataset):
        '''
        example BERT
        '''
        outputs = trainer.predict(dataset)
        with torch.no_grad():
            logits = outputs.predictions
            probabilities = F.softmax(torch.tensor(logits), dim=-1)
            predictions = logits.argmax(axis=-1)
        return {
            # 'probabilities': torch.max(probabilities, dim=-1).values.cpu().tolist(),
            'scores': probabilities[:,-1].tolist(),
            'y_pred': predictions.tolist(),
            'y_true': outputs.label_ids,
        }
    
    def predict(self, dataset, prompt, label_to_rank):
        # prepare inputs
        grouped_inputs = defaultdict(list)
        for row in dataset:
            test_str = prompt.format(row['text'], "")
            tokenized_input = self.tokenizer(
                test_str,
                return_tensors="pt",
                add_special_tokens=False
            )
            length = tokenized_input['input_ids'].shape[1]
            # input_ids, attention_mask, text, label
            item = (tokenized_input['input_ids'], tokenized_input['attention_mask'], test_str, row['label'])
            grouped_inputs[length].append(item)

        # feed model by batch
        batch_size = 4
        all_probs = []
        all_predicts, all_predictions = [], []
        all_strings = []
        all_labels = []
        for length, group in tqdm(grouped_inputs.items()):
            for i in range(0, len(group), batch_size):
                batch = group[i:i+batch_size]
                input_ids, attention_mask, batch_texts, batch_labels = zip(*batch)

                # Concatenate the batch inputs
                input_ids = torch.cat(input_ids, dim=0).to('cuda')
                attention_mask = torch.cat(attention_mask, dim=0).to('cuda')
                with torch.no_grad():
                    # Forward pass
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    # print(outputs.logits[:, -1].shape)
                
                # logits are shape (batch_size, sequence_length, num_classes)
                # we want only the last token of each sequence in the batch
                logits = outputs.logits[:, -1, :]
                
                # Apply softmax
                probabilities = F.softmax(logits, dim=-1)
                probs = torch.max(probabilities, dim=-1).values.cpu().tolist()
                all_probs.extend(probs)
                predictions = torch.argmax(probabilities, dim=-1)
                predictions = [self.tokenizer.decode(i).title() for i in predictions.cpu()]
                all_predicts.extend(predictions)

                predictions = [label_to_rank.get(i) for i in predictions]
                all_predictions.extend(predictions)

                all_labels.extend(batch_labels)
                all_strings.extend(batch_texts)

        return pd.DataFrame({
            # max prob
            'probabilities': all_probs,
            # predicted labels
            'predicts': all_predicts,
            # digits of labels
            'y_pred': all_predictions,
            # labels
            'y_true': all_labels,
        })
    
