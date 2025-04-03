import json
import os
from copy import deepcopy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout
# ROC curve only for binary classfication
from sklearn.metrics import roc_curve, auc

class MyAnn:
    def __init__(self):
        self.epochs = 10
        self.model = None
    
    def get_xy(self, df):
        self.X = np.array(df.iloc[:,2:], dtype=np.float16)
        self.y = np.array(df.iloc[:,1], dtype=np.float16)
        print('X:', self.X.shape, self.X.dtype)
        print('y:', self.y.shape, self.y.dtype)
        print('labels:', Counter(self.y))

        # normalization X
        scaler = StandardScaler()
        norm_X = scaler.fit_transform(self.X)

        #split data into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            norm_X, self.y, train_size=0.8, shuffle=True, random_state=2
        )
        self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(
            self.X_train, self.y_train, train_size=0.7, shuffle=True, random_state=2
        )
        print('train data:', self.X_train.shape, self.y_train.shape)
        print('validate data:', self.X_validate.shape, self.y_validate.shape)
        print('test data', self.X_test.shape, self.y_test.shape)
    
    def declare_model(self):
        # input features
        num_features = int(self.X.shape[-1])
        # outccome is binary
        self.model = tf.keras.Sequential([
            Input(shape=(num_features,)),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(96, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid'),
        ])
        print(self.model.summary())
    
    def train(self, epochs=None):
        self.epochs = 10 if epochs is None else epochs 
        # loss_fn = tf.keras.losses.sparse_categorical_crossentropy
        optim = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=optim,
            metrics=['accuracy', 'precision', 'recall', 'auc', 'mse']
        )
        # train
        self.history= self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_validate, self.y_validate),
            epochs=self.epochs,
            verbose=True
        )
        stat = pd.DataFrame(self.history.history)
        print(stat.head(3))
        return stat

    def test(self):
        '''
        evaluate model using test data
        '''
        eval_res = self.model.evaluate(
            self.X_test,
            self.y_test,
            return_dict=True
        )
        print(eval_res)
        # 
        self.pred_test = self.model.predict(self.X_test)
        print(self.pred_test[:3])
        return self.pred_test


    def plot_acc_recall(self):
        '''
        draw plots of accuracy, recall, and loss
        '''
        fig, axes = plt.subplots(1, 3, figsize=(12,3), layout='tight')
        x = range(self.epochs)
        fig.supxlabel('epochs')

        i=0
        axes[i].plot(x, self.history.history['recall'], label='recall of train')
        axes[i].plot(x, self.history.history['val_recall'], label='recall of validation')
        axes[i].set_ylabel('TP/(TP+FN) %')
        axes[i].set_title('Recall')
        axes[i].legend(loc='lower right', fontsize=8)

        i=1
        axes[i].plot(x, self.history.history['accuracy'], label='accurarcy of train')
        axes[i].plot(x, self.history.history['val_accuracy'], label='accurarcy of validation')
        axes[i].set_ylabel('(TP+TN)/N %')
        axes[i].set_title('Accuracy')
        axes[i].legend(loc='lower right', fontsize=8)

        i=2
        axes[i].plot(x, self.history.history['loss'], label='loss of train')
        axes[i].plot(x, self.history.history['val_loss'], label='loss of validation')
        axes[i].set_ylabel('Loss')
        axes[i].set_title('Loss')
        axes[i].legend(loc='upper right', fontsize=8)

    def plot_prob(self):
        # draw plot of probability
        t = self.pred_test[self.y_test==1]
        f = self.pred_test[self.y_test==0]
        fig, ax = plt.subplots(1, 2, figsize=(8,3), layout='tight')
        fig.supxlabel('probability')
        fig.supylabel('number')

        ax[0].hist(t, label='True', bins=20)
        ax[0].set_title(f'Prediction of epitopes {len(t)}')
        ax[1].hist(f, label='False', bins=20)
        ax[1].set_title(f'Prediction of non-epitopes {len(f)}')

    def plot_roc(self):
        # # FPR, false positive rate, 1- specificity, typ I error
        # TPR, true positive rate, sensitivity, power
        fpr, tpr, thresholds = roc_curve(self.y_test, self.pred_test)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(1, figsize=(4,3))
        ax.plot(fpr, tpr)
        ax.set_xlabel('1-specificity (False positive rate)')
        ax.set_ylabel('sensitivity (True positive rate)')
        plt.title(f'ROC (area={roc_auc:.2f})')
        ax.plot([0,1], [0,1], '--')
        ax.plot([0,0,1], [0,1,1], ':')
        
    def save_model(self, outfile):
        self.model.save(outfile)
        return self.model