
import numpy as np
from sklearn.naive_bayes import  MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.model_selection import KFold
from utils import get_prediction , get_prediction_svm
import operator

class Classifier(object):

    def __init__(self, classifier, x_train, y_train, x_test):
        self.classifier = classifier
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

    def predict(self):
        if self.classifier == 'naive_bayes':
            return self.naiveBayesClassifier()
        elif self.classifier == 'svm':
            return self.svmClassifier()
        elif self.classifier == 'dt':
            return self.decisionTreeClassifier()
        else:
            return 'Classifier en construccion ...'

    def naiveBayesClassifier(self):
        clf = MultinomialNB()
        clf.fit(self.x_train, self.y_train)####

        #prediction = clf.predict(self.x_test)
        proba = clf.predict_proba(self.x_test)
        predictions = get_prediction(proba, clf.classes_)

        return predictions

    def svmClassifier(self):
        clf = LinearSVC()
        #clf = SVC(probability=True)
        clf.fit(self.x_train, self.y_train)
        prediction = clf.predict(self.x_test)
        #proba = clf.predict_proba(self.x_test)  ### en veremos, probablemente no
        distances = clf.decision_function(self.x_test)
        predictions = get_prediction_svm(distances, clf.classes_)
        return predictions

    def decisionTreeClassifier(self):
        clf = tree.DecisionTreeClassifier(max_depth=5)
        clf.fit(self.x_train, self.y_train)
        #prediction = clf.predict(x_test)
        proba = clf.predict_proba(self.x_test)
        predictions = get_prediction(proba, clf.classes_)
        return predictions


class KFoldCrossValidation(object):

    def __init__(self, features, labels, classifier='naive_bayes', k=10):
        self.features = features
        self.labels = labels
        self.classifier = classifier
        self.kfolds = k

    def train_and_predict(self):
        kf = KFold(n_splits=self.kfolds)
        all_predictions = []

        for train_index, test_index in kf.split(self.features):
            x_train = self.features[train_index]
            x_test = self.features[test_index]

            y_train = self.labels[train_index]
            y_test = self.labels[test_index]

            obj = Classifier(self.classifier, x_train, y_train, x_test)
            prediction = obj.predict()
            all_predictions.extend(prediction)

        return all_predictions



def habers():
    x = np.array([[10, 20, 25], [30, 40, 23], [11, 21, 5], [32, 42, 4], [50, 60, 70], [70, 70, 12], [3, 5, 10], [1,1,2], [3,3,5], [10,10,11]])
    y = np.array([True, False, False, False, True, True, False, True, False, True])
    kf = KFold(n_splits=3)
    index_predictions = []
    predictions = []
    for train_index, test_index in kf.split(x):
        x_train = x[train_index]
        x_test = x[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        #print 'x_train: ' , x_train
        #print 'x_test: ', x_test
        #print 'y_train: ', y_train
        #print 'y_test: ' , y_test
        #print ''

        #clf = MultinomialNB()
        #clf = LinearSVC(probability=True)
        #clf = SVC(probability=True)
        clf = tree.DecisionTreeClassifier()

        clf.fit(x_train, y_train)

        prediction = clf.predict(x_test)
        proba = clf.predict_proba(x_test)
        #distances = clf.decision_function(x_test)
        print 'x test: ',  x_test
        #print 'predicted/real: ' , prediction, y_test
        print 'predicted: ', prediction
        print 'probabilities: ' , proba
        #print 'distances: ' , distances
        #print clf.classes_  # False True
        print ''

        index_predictions.extend(test_index)
        predictions.extend(prediction)

    print index_predictions
    print predictions




if __name__ == '__main__':

    features = np.array([[10, 20, 25], [30, 40, 23], [11, 21, 5], [32, 42, 4], [50, 60, 70], [70, 70, 12], [3, 5, 10], [1,1,2], [3,3,5], [10,10,11]])
    labels = np.array([False, False, False, False, True, True, False, True, False, True])
    classifier = 'naive_bayes'  # 'svm'  'dt'  'naive_bayes'

    obj = KFoldCrossValidation(features, labels, classifier=classifier ,k=5)
    predictions = obj.train_and_predict()

    for i in predictions:
        print i


    #lenghts = [3,2,5]

    #print list_split(predictions, lenghts)








