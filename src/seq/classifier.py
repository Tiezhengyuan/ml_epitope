import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class Classifier:
    def __init__(self):
        self.df = None
    
    def get_df(self, infile, part:bool=False) -> pd.DataFrame:
        df = pd.read_csv(infile, sep='\t', header=0, index_col=None)
        # balance the number of epitopes and non-epitopes
        # shuffle rows
        df = df.sample(frac=1)
        # trim data
        if part is True:
            df = df.iloc[:10_000,:]
        # add label
        df['label'] = df['label'].map({'epitope':1, 'other': 0})
        print('input data:', df.shape)
        return df
    
    def xy(self, df):
        '''
        prepare y ~ X
        '''
        # get X and y
        X = np.array(df.iloc[:,:-1], dtype=np.float16)
        y = np.array(df.iloc[:,-1], dtype=np.float16)

        # normalization X
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        # output
        print('X and y:', X.shape, X.dtype, y.shape, y.dtype)
        print('labels:', Counter(y))

        #split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.7, shuffle=True, random_state=2
        )
        print('train data:', X_train.shape, y_train.shape)
        print('test_data: ', X_test.shape, y_test.shape)
        return X_train, X_test, y_train, y_test
    
    def train_rf(self, df):
        '''
        random forest model
        '''
        # prepare X and y
        X_train, X_test, y_train, y_test = self.xy(df)
        
        # RF model
        model = RandomForestClassifier(
            n_estimators=500,
            max_leaf_nodes=16,
            n_jobs=-1,
            random_state=42
        )
        # train
        model.fit(X_train, y_train)
        return model, X_test, y_test