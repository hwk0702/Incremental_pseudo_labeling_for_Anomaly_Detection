from sklearn.svm import OneClassSVM
import joblib


class OCSVM(OneClassSVM):

    def __init__(self, kernel, gamma):
        self.kernel = kernel
        self.gamma = gamma
        self.OS = OneClassSVM(kernel=self.kernel, gamma=self.gamma)

    def model_reset(self):
        self.OS = OneClassSVM(kernel=self.kernel, gamma=self.gamma)

    def train(self, x_train):
        self.OS = self.OS.fit(x_train)

    def validation(self, x_val):
        result = -self.OS.decision_function(x_val)
        return result

    def test(self, x_test):
        result = -self.OS.decision_function(x_test)
        return result

    def save(self, filename):
        joblib.dump(self.OS, f'{filename}.pkl')

    def load(self, path):
        self.OS = joblib.load(path)



if __name__ == '__main__':
    pass