import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


def train_val_test_split(X, y, splitCount, unlabelSize, test_val_size, random_seed):
    split = StratifiedShuffleSplit(n_splits=1, test_size=unlabelSize, random_state=random_seed)

    train = []
    unlabel = []
    label = []
    test_val = []
    test = []
    val = []

    for label_id, unlabel_id in split.split(X, y):

        label.append(label_id)
        unlabel.append(unlabel_id)

        label_X = X.iloc[label[0]]
        label_y = y.iloc[label[0]]

        unlabel_X = X.iloc[unlabel[0]]
        unlabel_y = y.iloc[unlabel[0]]

        split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_val_size, random_state=random_seed)

        for train_id, test_val_id in split2.split(label_X, label_y):

            train.append(train_id)
            test_val.append(test_val_id)

            test_val_X = label_X.iloc[test_val[0]]
            test_val_y = label_y.iloc[test_val[0]]

            train_X = label_X.iloc[train[0]]
            train_y = label_y.iloc[train[0]]

            split3 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_seed)

            for val_id, test_id in split3.split(test_val_X, test_val_y):
                val.append(val_id)
                test.append(test_id)

                val_X = test_val_X.iloc[val[0]]
                val_y = test_val_y.iloc[val[0]]

                test_X = test_val_X.iloc[test[0]]
                test_y = test_val_y.iloc[test[0]]

    return train_X, train_y, unlabel_X, unlabel_y, val_X, val_y, test_X, test_y


def train_val_test_split2(X, y, test_val_size, unlabelSize, random_seed):    
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_val_size, random_state = random_seed)
    
    test_val=[]
    data = []    
    train =[]
    unlabel =[]
    label=[]    
    test =[]
    val =[]
    
    for data_id, test_val_id in split.split(X,y):
        
        data.append(data_id)        
        test_val.append(test_val_id)
        
        data_X = X.iloc[data[0]]
        data_y = y.iloc[data[0]]
                
        test_val_X = X.iloc[test_val[0]]
        test_val_y = y.iloc[test_val[0]]
        
        split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state =random_seed)    
        for test_id, val_id in split2.split(test_val_X, test_val_y):
            test.append(test_id)
            val.append(val_id)
            
            test_X = test_val_X.iloc[test[0]]
            test_y = test_val_y.iloc[test[0]]
            
            val_X = test_val_X.iloc[val[0]]
            val_y = test_val_y.iloc[val[0]]            
            
            
        split3 = StratifiedShuffleSplit(n_splits=1, test_size=unlabelSize, random_state=random_seed)        
        for train_id, unlabel_id in split3.split(data_X, data_y):
            
            train.append(train_id)
            unlabel.append(unlabel_id)
            
            train_X = data_X.iloc[train[0]]            
            train_y = data_y.iloc[train[0]]
            
            unlabel_X = data_X.iloc[unlabel[0]]
            unlabel_y = data_y.iloc[unlabel[0]]
        
        return  train_X, train_y, unlabel_X, unlabel_y, val_X, val_y, test_X, test_y

def getTestValIndex(y, random_seed):
    # Anomal label
    anomalLabel = 1

    isAnomal = y == anomalLabel
    isNormal = y != anomalLabel

    anomal = y[isAnomal].index.tolist()
    normal = y[isNormal].index.tolist()

    sampledNormal = y.loc[normal].sample(n=len(anomal), random_state=random_seed).index.tolist()

    return y[anomal + sampledNormal].index.tolist()


def getDatasets(k, X, y):
    try:
        assert (k < 20), f"k must be 0~19"
    except AssertionError as e:
        raise
    data = train_val_test_split2(X, y, test_val_size=0.3,  unlabelSize=0.5, random_seed=k+72)

    trainX = data[0].reset_index(drop=True)
    trainy = data[1].reset_index(drop=True)
    unkX = data[2].reset_index(drop=True)
    unky = data[3].reset_index(drop=True)
    valX = data[4].reset_index(drop=True)
    valy = data[5].reset_index(drop=True)
    idx = getTestValIndex(valy, k)
    valX = valX.iloc[idx]
    valy = valy.iloc[idx]
    testX = data[6].reset_index(drop=True)
    testy = data[7].reset_index(drop=True)
    idx = getTestValIndex(testy, k)
    testX = testX.iloc[idx]
    testy = testy.iloc[idx]

    return {'x_train': trainX, 'y_train': trainy,
            'x_unk': unkX, 'y_unk': unky,
            'x_test': testX, 'y_test': testy,
            'x_val': valX, 'y_val': valy}

if __name__ == '__main__':
    pass