import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTree
import time

class Bagging:
    
    def __init__(self, n_iter=20):
        self.n_iter = n_iter
        self.estimator = DTree(random_state=1)
        
    def train(self, X_train, y_train,percentage=0.2, feature_names=None):
        self.percentage = percentage
        self.estimators = []
        n_samples = X_train.shape[0]
        start_time = time.time()
        for i in range(self.n_iter):
            indices = np.random.randint(0, n_samples, np.int(self.percentage * n_samples))
            cur_X_train = X_train[indices]
            cur_y_train = y_train[indices]
            self.estimator.fit(cur_X_train, cur_y_train)
            self.estimators.append(self.estimator)
        end_time = time.time()
        return end_time - start_time
    
    def predict(self, X_test, y_test = None):
        predctions = []
        for x in X_test:
            cur_prediction = Counter()
            for estimator in self.estimators:
                cur_prediction[estimator.predict(x.reshape(1,len(x)))[0]] += 1
            predctions.append(cur_prediction.most_common(1)[0][0])
        predctions = np.array(predctions)
        if y_test is not None:
            accuracy = np.sum(predctions == y_test)/len(y_test)
            print('Accuracy:', accuracy)
            return predctions, accuracy
        return predctions
        