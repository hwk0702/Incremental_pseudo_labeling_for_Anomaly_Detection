from sklearn.ensemble import IsolationForest
import joblib


class IF(IsolationForest):

    def __init__(self, n_estimators, max_samples, contamination, random_state):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.IF = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                  contamination=self.contamination, random_state=self.random_state)

    def model_reset(self):
        self.IF = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                  contamination=self.contamination, random_state=self.random_state)

    def train(self, x_train):
        self.IF = self.IF.fit(x_train)

    def validation(self, x_val):
        # result = self.IF.decision_function(x_val)
        result = -self.IF.score_samples(x_val)
        return result

    def test(self, x_test):
        # result = self.IF.decision_function(x_test)
        result = -self.IF.score_samples(x_test)
        return result

    def save(self, filename):
        joblib.dump(self.IF, f'{filename}.pkl')

    def load(self, path):
        self.IF = joblib.load(path)


if __name__ == '__main__':
    pass