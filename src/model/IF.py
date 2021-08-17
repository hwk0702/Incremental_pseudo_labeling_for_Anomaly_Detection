from sklearn.ensemble import IsolationForest
import joblib


class IF(IsolationForest):

    def __init__(self, model_params):
        self.model_params = model_params
        self.IF = IsolationForest(**self.model_params)

    def model_reset(self):
        self.IF = IsolationForest(**self.model_params)

    def train(self, x_train):
        self.IF = self.IF.fit(x_train)

    def validation(self, x_val):
        # result = self.IF.decision_function(x_val)
        result = self.IF.score_samples(x_val)
        return result

    def test(self, x_test):
        # result = self.IF.decision_function(x_test)
        result = self.IF.score_samples(x_test)
        return result

    def save(self, filename):
        joblib.dump(self.IF, f'{filename}.pkl')

    def load(self, path):
        self.IF = joblib.load(path)


if __name__ == '__main__':
    pass