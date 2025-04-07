import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score,\
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,\
    precision_recall_curve, roc_curve, roc_auc_score

class TrainerPlot:

    @staticmethod
    def loss(trainer, results_dir=None):
        hist = pd.DataFrame(trainer.state.log_history)
        # plot
        plt.figure(figsize=(6,3))
        plt.scatter(hist['step'], hist['loss'])
        plt.title('Loss in traning process')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        if results_dir:
            outfile = os.path.join(results_dir, 'train_hist.csv')
            hist.to_csv(outfile)

    @staticmethod
    def scores(stat):
        stat = stat.dropna()
        print('Total: ', len(stat['y_true']))
        print('true classifications:', Counter(stat['y_true']))
        print('predictions: ', Counter(stat['y_pred']))

        # precision and recall
        precision = precision_score(stat['y_true'], stat['y_pred'])
        recall = recall_score(stat['y_true'], stat['y_pred'])
        accuracy = accuracy_score(stat['y_true'], stat['y_pred'])
        f1 = f1_score(stat['y_true'], stat['y_pred'])

        return {
            'recall': recall,
            'precision': precision,
            'F1 score': f1,
            'accuracy': accuracy,
        }

    @staticmethod
    def cm(stat, keys:tuple=None):
        stat = stat.dropna()
        k1, k2 = ('y_true', 'y_pred') if keys is None else keys
        # confusion matrix
        cm = confusion_matrix(stat[k1], stat[k2])
        print('confusion matrix:\n', cm)
        ConfusionMatrixDisplay.from_predictions(
            stat[k1], stat[k2]
        )

    @staticmethod    
    def recall_roc(stat, threshold:int=.5):
        # draw plots
        fig, ax=plt.subplots(1,2, figsize=(9,3))

        precisions, recalls, thresholds = precision_recall_curve(
            stat['y_true'], stat['scores']
        )
        i=0
        ax[i].plot(thresholds, precisions[:-1], linewidth=2, label='precision')
        ax[i].plot(thresholds, recalls[:-1], linewidth=2, label='recall')
        ax[i].vlines(threshold, 0, 1, 'k', 'dotted')
        ax[i].set_title('Precision and Recall')
        ax[i].set_xlabel('threshold: Probability')
        ax[i].set_ylabel('Percentage')
        ax[i].legend(loc='lower left')

        # ROC
        fpr, tpr, thresholds = roc_curve(
            stat['y_true'], stat['scores']
        )
        auc = roc_auc_score(stat['y_true'], stat['scores'])
        i=1
        ax[i].plot(fpr, tpr, linewidth=2, label='test')
        ax[i].plot([0,1], [0,1], 'k:')
        ax[i].set_title('ROC curve')
        ax[i].set_xlabel('False positive rate (fall-out)')
        ax[i].set_ylabel('True positive rate (recall)')
        ax[i].text(0.6, 0.5, f"AUC={auc:.2f}")
        plt.show()

        return {
            'AUC': auc,
        }

