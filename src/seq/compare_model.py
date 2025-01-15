import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from collections import Counter
from sklearn.preprocessing import StandardScaler

class CompareModel:
    def __init__(self, model1, model2):
        self.m1 = model1
        self.m2 = model2

    def predict(self, df):
        self.X = np.array(df.iloc[:,2:], dtype=np.float16)
        self.y = np.array(df.iloc[:,1], dtype=np.float16)
        print('X:', self.X.shape, self.X.dtype)
        print('y:', self.y.shape, self.y.dtype)
        print('labels:', Counter(self.y))

        # normalization X
        scaler = StandardScaler()
        norm_X = scaler.fit_transform(self.X)

        # predict
        self.pred1 = self.m1.predict(norm_X)
        self.pred2 = self.m2.predict(norm_X)

    def plot_roc(self, name1, name2):
        # # FPR, false positive rate, 1- specificity, typ I error
        # TPR, true positive rate, sensitivity, power
        fpr1, tpr1, thresholds1 = roc_curve(
            self.y, self.pred1
        )
        roc_auc1 = auc(fpr1, tpr1)
        fpr2, tpr2, thresholds2 = roc_curve(
            self.y, self.pred2
        )
        roc_auc2 = auc(fpr2, tpr2)

        fig, ax = plt.subplots(1, figsize=(4,3))
        ax.plot(fpr1, tpr1, label=f'{name1} area={roc_auc1:.2f}')
        ax.plot(fpr2, tpr2, label=f'{name2} area={roc_auc2:.2f}')
        ax.set_xlabel('1-specificity (False positive rate)')
        ax.set_ylabel('sensitivity (True positive rate)')
        plt.title(f'ROC of the model {name1} and {name2}')
        ax.plot([0,1], [0,1], '--')
        ax.plot([0,0,1], [0,1,1], ':')