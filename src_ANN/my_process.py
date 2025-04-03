import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

from .my_dataset import MyDataset


class MyProcess:
    loss_fn = nn.BCELoss()
    batch_size = 32

    def __init__(self, model, collate_fn, num_epochs=None):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.collate_fn = collate_fn
        self.num_epochs = 5 if num_epochs is None else num_epochs
    
    def run(self, train_ds, valid_ds) -> tuple:
        info = pd.DataFrame(
            np.zeros((self.num_epochs, 5), dtype=float),
            columns= ['acc_train', 'loss_train', 'acc_valid', 'loss_valid', 'rec_valid']
        )

        # build dataloader, convert text into vector
        # batching the datasets. shuffle data for each epoch
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, \
            shuffle=True, collate_fn=self.collate_fn)
        valid_dl = DataLoader(valid_ds, batch_size=self.batch_size, \
            shuffle=True, collate_fn=self.collate_fn)

        for epoch in range(self.num_epochs):
            # training
            acc_train, loss_train = self.train(train_dl)
            # validation
            acc_valid, loss_valid, rec_valid = self.evaluate(valid_dl)
            # output of one epoch
            print(f'Epoch {epoch} acc: {acc_train:.4f} val_acc: {acc_valid:.4f}, val_recall: {rec_valid:.4f}')
            info.loc[epoch] = [acc_train, loss_train, acc_valid, loss_valid, rec_valid]
        return self.model, info

    def train(self, dataloader):
        '''
        train model
        '''
        self.model.train()
        total_acc, total_loss, total_recall = 0, 0, 0
        for input_batch in dataloader:
            self.optimizer.zero_grad()
            pred = self.model(input_batch)[:, 0]
            label_batch = input_batch['labels']
            loss = self.loss_fn(pred, label_batch)
            loss.backward()
            self.optimizer.step()
            # true positive and true negative
            total_acc += ((pred>=0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item()*label_batch.size(0)
        acc = total_acc/len(dataloader.dataset)
        loss = total_loss/len(dataloader.dataset)
        return acc, loss

    def evaluate(self, dataloader) -> tuple:
        '''
        validate model
        '''
        self.model.eval()
        total_acc, total_loss = 0, 0
        np_target, np_pred = np.array([]), np.array([])
        with torch.no_grad():
            for input_batch in dataloader:
                pred = self.model(input_batch)[:, 0]
                label_batch = input_batch['labels']
                loss = self.loss_fn(pred, label_batch)
                total_acc += ((pred>=0.5).float() == label_batch).float().sum().item()
                total_loss += loss.item()*label_batch.size(0)
                # 
                _np_target = label_batch.view(-1).cpu()
                _np_pred = pred.view(-1).cpu()
                _np_pred = (_np_pred >= .5).numpy().astype(int)
                np_target = np.concatenate([np_target, _np_target])
                np_pred = np.concatenate([np_pred, _np_pred])
        recall = recall_score(np_target, np_pred, average = 'macro', zero_division=np.nan)
        acc = total_acc/len(dataloader.dataset)
        loss = total_loss/len(dataloader.dataset)
        return acc, loss, recall

    def test(self, test_ds:list) -> dict:
        dl = DataLoader(test_ds, batch_size=self.batch_size,
            shuffle=False, collate_fn=self.collate_fn)
        acc, loss, rec = self.evaluate(dl)
        return {'recall': rec, 'accuracy': acc, 'loss': loss}

    def predict(self, texts:list) -> tuple:
        '''
        predictor
        '''
        total_pred = np.array([])
        labels = [0] * len(texts)
        
        dataset = MyDataset(texts, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                shuffle=False, collate_fn=self.collate_fn)
        self.model.eval()
        with torch.no_grad():
            for input_batch in dataloader:
                pred = self.model(input_batch)[:, 0]
                _pred = pred.view(-1).cpu().numpy()
                total_pred = np.concatenate([total_pred, _pred])
        return pd.DataFrame({'text':texts, 'predict': list(total_pred)})
