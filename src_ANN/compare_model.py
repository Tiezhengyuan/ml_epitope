from collections import Counter
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

class CompareModel:

    @staticmethod
    def predict(df, model_path):
        X = np.array(df.iloc[:,2:], dtype=np.float16)
        y = np.array(df.iloc[:,1], dtype=np.float16)
        print('X:', X.shape, X.dtype)
        print('y:', y.shape, y.dtype)
        print('labels:', Counter(y))

        # normalization X
        scaler = StandardScaler()
        norm_X = scaler.fit_transform(X)

        # predict
        model = tf.keras.models.load_model(model_path)
        pred = model.predict(norm_X)
        return y, pred

    @staticmethod
    def plot_roc(m1, m2):
        # # FPR, false positive rate, 1- specificity, typ I error
        # TPR, true positive rate, sensitivity, power
        name1, name2 = m1.name, m2.name
        fpr1, tpr1, thresholds1 = roc_curve(
            m1['y'], m1['pred']
        )
        roc_auc1 = auc(fpr1, tpr1)
        fpr2, tpr2, thresholds2 = roc_curve(
            m2['y'], m2['pred']
        )
        roc_auc2 = auc(fpr2, tpr2)

        fig, ax = plt.subplots(1, figsize=(4,3))
        fig.suptitle(f"ROC of the model {name1} vs {name2}")
        ax.plot(fpr1, tpr1, label=f"{name1} area={roc_auc1:.2f}")
        ax.plot(fpr2, tpr2, label=f"{name2} area={roc_auc2:.2f}")
        ax.set_xlabel('1-specificity (False positive rate)')
        ax.set_ylabel('sensitivity (True positive rate)')
        ax.legend()
        ax.plot([0,1], [0,1], '--')
        ax.plot([0,0,1], [0,1,1], ':')
    
    @staticmethod
    def plot_hist(m1, m2):
        # Plot histograms
        fig, ax = plt.subplots(1,2, figsize=(10,4), layout='tight')
        fig.supxlabel('Probability')
        fig.supylabel('Frequency')
        fig.suptitle(f'Comparison of {m1.name} vs {m2.name}')

        i=0
        ax[i].hist(m1[m1['y']==1]['pred'], bins=50, alpha=0.5, label=m1.name)
        ax[i].hist(m2[m2['y']==1]['pred'], bins=50, alpha=0.5, label=m2.name)
        ax[i].legend(loc='upper left')
        ax[i].set_title('predictions of epitopes')

        i=1
        ax[i].hist(m1[m1['y']==0]['pred'], bins=50, alpha=0.5, label=m1.name)
        ax[i].hist(m2[m2['y']==0]['pred'], bins=50, alpha=0.5, label=m2.name)
        ax[i].legend(loc='upper left')
        ax[i].set_title('predictions of non-epitopes')

        # Show plot
        plt.show()