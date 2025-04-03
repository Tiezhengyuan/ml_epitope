import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class MyEvaluate:

    @staticmethod
    def plot_acc(stat):
        epochs = len(stat)
        x_arr = list(range(epochs))

        fig = plt.figure(figsize=(10,3))
        ax = fig.add_subplot(1,2,1)
        ax.plot(x_arr, stat.acc_train, label="Accuracy of train")
        ax.plot(x_arr, stat.acc_valid, label="Accuracy of validation")
        ax.plot(x_arr, stat.rec_valid, label="Recall of validation")
        ax.legend(fontsize=12)
        # ax.set_ylim(.8, 1)
        ax.set_title('Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Percentage %')

        ax = fig.add_subplot(1,2,2)
        ax.plot(x_arr, stat.loss_train, label="Loss of train")
        ax.plot(x_arr, stat.loss_valid, label="Loss of validation")
        ax.legend(fontsize=12)
        ax.set_title('Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')

        plt.show()
    
    @staticmethod
    def plot_prob(pred_test):
        t = pred_test[pred_test['labels']=='epitope']['predict']
        f = pred_test[~(pred_test['labels']=='epitope')]['predict']
        print(len(t), len(f))

        fig, ax = plt.subplots(1, 2, figsize=(9,3), layout='tight')
        fig.suptitle('Evavluate RNN model using test data')
        fig.supxlabel('probability')
        fig.supylabel('number')

        ax[0].hist(t, bins=50)
        ax[0].set_title(f'Predict epitopes, n = {len(t)}')
        ax[1].hist(f, bins=50)
        ax[1].set_title(f'Distinguish non-epitopes, n = {len(f)}')
    
    @staticmethod
    def plot_roc(pred_test):
        plt.figure(figsize=(5,3))
        fpr, tpr, thresholds = roc_curve(pred_test['labels'], pred_test['predict'], pos_label='epitope')
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='black')
        plt.title(f'ROC (area={roc_auc:.2f})')
        plt.xlabel('False postive rate')
        plt.ylabel('True postive rate')
        plt.plot([0,1],[0,1], linestyle='--', color='grey')
        plt.plot([0,0,1],[0,1,1], linestyle=':', color='grey')