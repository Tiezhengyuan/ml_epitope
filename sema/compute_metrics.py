import scipy
from transformers import EvalPrediction
from sklearn.metrics import r2_score, mean_squared_error

class ComputeMetrics:

    @staticmethod
    def text_regression(p: EvalPrediction):
        
        preds = p.predictions[:,:,1]

        batch_size, seq_len = preds.shape    
        out_labels, out_preds = [], []

        for i in range(batch_size):
            for j in range(seq_len):
                if p.label_ids[i, j] > -1:
                    out_labels.append(p.label_ids[i][j])
                    out_preds.append(preds[i][j])
                    
        out_labels_regr = out_labels#[math.log(t+1) for t in out_labels]

        
        return {
            "pearson_r": scipy.stats.pearsonr(out_labels_regr, out_preds)[0],
            "mse": mean_squared_error(out_labels_regr, out_preds),
            "r2_score": r2_score(out_labels_regr, out_preds)
        }