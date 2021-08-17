"""
train.py
Autor: JungwooChoi, HyeongwonKang
Incremental_pseudo_labeling_for_Anomaly_Detection
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model import AE, GAN, IF, OCSVM
import warnings
import collections

warnings.filterwarnings('ignore')


class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print(f'Training process is stopped early....')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False

def train(model_name, X, Y, oversampling=None, n_estimator=None, kernel_size = None,
          params_name=None, epochs=100, batch_size=50, lossType=None):
    """
    model training

    Arguments
    ---------
    model_name: model name (RandomForestClassifier, XGBClassifier, LGBMClassifier)
    X: data
    Y: label
    n_estimator: machine learning num of estimator
    params_name: model parameter

    """
    global model
    if model_name in ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier']:
        model = ML_train(model_name, X, Y, n_estimator, **params_name)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        early_stopping = EarlyStopping(patience=500, verbose=1)
        if lossType == 'focal':
            if oversampling != None:
                alpha = float(collections.Counter(Y)[1]/len(Y))
            else:
                alpha = 0.5
            criterion = WeightedFocalLoss(alpha=alpha)
        else:
            criterion = nn.BCEWithLogitsLoss()

        input_channel = X.shape[2]
        model = CNN(input_channel, kernel_size).to(device)
        optimizer = optim.Adam(
            model.parameters(), lr=0.001  # , momentum=0.9, weight_decay=1e-3
        )

        X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

        ds = TensorDataset(X, Y)

        if oversampling == 'WeightedSampler':
            dataloader = DataLoader(ds, sampler=ImbalancedDatasetSampler(ds), batch_size=batch_size, shuffle=True)
        else:
            dataloader = DataLoader(ds, batch_size=batch_size)

        min_loss = float("inf")
        state_dict={}

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            for batch_x, batch_y in dataloader:
                # print(batch_x.shape)
                #                 print(batch_x.unsqueeze(1), batch_y)
                #                 print(batch_x.unsqueeze(1).shape, batch_y.shape)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                mo = model(batch_x)
                loss = criterion(mo, batch_y)
                loss.backward()
                epoch_loss += float(loss.clone().detach().cpu())
                acc = binary_acc(mo, batch_y)
                epoch_acc += acc
                optimizer.step()
            print("epoch:", epoch, "- loss:", epoch_loss/len(dataloader), " acc:", epoch_acc/len(dataloader))
            if epoch_loss < min_loss:
                state_dict = model.state_dict()
                min_loss = epoch_loss
            if early_stopping.validate(epoch_loss):
                break
                # torch.save(model.state_dict(), PATH)
        model.load_state_dict(state_dict)

    return model


def predict(model, model_name, X_test):
    """
    model predict

    Arguments
    ---------
    model: trained model
    model_name: model name (RandomForestClassifier, XGBClassifier, LGBMClassifier)
    X_test: test data

    """
    global Y_pred
    if model_name in ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier']:
        Y_pred = model.predict(X_test)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
        Y_pred = torch.round(torch.sigmoid(model(X_test))).detach().cpu().numpy()
    return Y_pred


def ML_train(model_name, X, Y, n_estimator: int, **params_name):
    """
    Machine Learning Model Training

    Arguments
    ---------
    model_name: model name (RandomForestClassifier, XGBClassifier, LGBMClassifier)
    X: data
    Y: label
    n_estimator: machine learning num of estimator
    params_name: model parameter

    """
    model = globals()[model_name](n_estimators=n_estimator).set_params(**params_name)
    model.fit(X, Y)
    return model


def binary_acc(y_pred, y_test):

    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    # print(y_pred_tag)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100).item()

    return acc

# def Conv1d_train(X, Y):



# def Conv1d_test():


if __name__ == '__main__':
    pass
