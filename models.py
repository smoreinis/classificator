import math, random, sys
import numpy as np

from time import time
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model.sparse import ElasticNet, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm.sparse import LinearSVC, NuSVC, SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.grid_search import GridSearchCV

class CombinedClassifier:
    def __init__(self, model_fn):
        self.model_fn = model_fn        
        self.name = self.clf = 'Combined ' + self.model_fn().name
        self.train_time, self.test_time = None, None
        self.label_set = None
        
    def train(self, data, labels):
        self.label_set = set(labels)
        self.title_clf, self.domain_clf = {}, {}
        t0 = time()
        for label in self.label_set:
            self.title_clf[label] = self.model_fn()
            self.title_clf[label].train(data[0], (labels == label).astype(int))
            self.domain_clf[label] = self.model_fn()
            self.domain_clf[label].train(data[1], (labels == label).astype(int))
        self.train_time = time() - t0
    
    def test_label(self, data):
        title_probs, domain_probs = {}, {}
        pred_labels, t0 = list(), time()
        for label in self.label_set:
            title_probs[label] = self.title_clf[label].test_prob(data[0])
            domain_probs[label] = self.domain_clf[label].test_prob(data[1])
            
        for i in range(data[0].shape[0]):
            score_list = list()
            for label in self.label_set:
                t_prob = title_probs[label][i]
                try:
                    t_prob = t_prob[1]
                except:
                    pass
                d_prob = domain_probs[label][i]
                try:
                    d_prob = d_prob[1]
                except:
                    pass
                score_list.append((t_prob * d_prob, label))
            #print score_list
            optVal, optLabel = max(score_list, key=lambda (u,v): u)
            pred_labels.append(optLabel)
        
        self.test_time = time() - t0
        return np.array(pred_labels)

class Classifier:
    def __init__(self, clf_model, params):
        self.clf = clf_model(**params)
        self.name = str(self.clf)
        self.train_time, self.test_time = None, None
        self.label_set = None
        
    def train(self, data, labels):
        self.label_set = set(labels)
        t0 = time()
        self.clf.fit(data, labels)
        self.train_time = time() - t0
        
    def test_label(self, data):
        t0 = time()
        pred = self.clf.predict(data)
        self.test_time = time() - t0
        return pred
        
    def test_score(self, data):
        prob_dict = {}
        t0 = time()
        for label in self.label_set:
            prob_dict[label] = self.clf.score(data, label)
        self.test_time = time() - t0
        return prob_dict
    
    def test_prob(self, data):
        try:
            t0 = time()
            probs = self.clf.predict_proba(data)
            self.test_time = time() - t0
            return probs
        except:
            print sys.exc_info()[0]
            print self.name, 'does not support probabilities'
        return None
        
def find_best_log(**params):
    parameters = {'C': [0.01, 0.1, 1, 10, 100, 1000]}
    return GridSearchCV(LogisticRegression(**params), parameters)
    
def find_best_ridge(**params):
    parameters = {'alpha': [0.01, 0.1, 1, 10]}
    return GridSearchCV(RidgeClassifier(**params), parameters)
    
def find_best_sgd(**params):
    parameters = {
        'alpha' : [0.0001, 0.0005, 0.001],
        'rho'   : [0.80, 0.85, 0.95],
    }
    return GridSearchCV(SGDClassifier(**params), parameters)
    
def find_best_lsvc(**params):
    parameters = {'C': [0.01, 0.1, 1, 10, 100, 1000]}
    return GridSearchCV(LinearSVC(**params), parameters)
    
def find_best_nsvc(**params):
    parameters = {
        'nu': [0.3, 0.4, 0.5, 0.6, 0.7],
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 0.5],
    }
    return GridSearchCV(NuSVC(**params), parameters)
    
def find_best_svc(**params):
    parameters = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 0.5],
    }
    return GridSearchCV(SVC(**params), parameters)

models = {
    #ElasticNet(alpha=1.0, rho=0.5, fit_intercept=False, normalize=False, max_iter=1000, tol=0.0001)
    'enet'  : (ElasticNet, 'Elastic Net Classifier'),
    #LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1)
    'log'   : (LogisticRegression, 'Logistic Regression Classifier'),
    #RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, overwrite_X=False, tol=0.001)
    'ridge' : (RidgeClassifier, 'Ridge Classifier'),
    #SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, rho=0.85, 
    #   fit_intercept=True, n_iter=5, shuffle=False, verbose=0, n_jobs=1, seed=0, learning_rate='optimal', eta0=0.0, power_t=0.5)
    'sgd'   : (SGDClassifier, 'SGD Classifier'),
    #BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True)
    'bnb'   : (BernoulliNB, 'Bernoulli NB Classifier'),
    #MultinomialNB(alpha=1.0, fit_prior=True)
    'mnb'   : (MultinomialNB, 'Multinomial NB Classifier'),
    #LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0, multi_class=False, fit_intercept=True, intercept_scaling=1)
    'lsvc'  : (LinearSVC, 'Linear SVC Classifier'),
    #NuSVC(nu=0.5, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001)
    'nsvc'  : (NuSVC, 'nu-SVC Classifier'),
    #SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001)
    'svc'   : (SVC, 'SVC Classifier'),
}
    
def make_classifier(type, combine=False, params=None):
    res = models.get(type, None)
    if res is None:
        print 'Cannot find model type', type
        return None
    clf_model, clf_name = res
    if params is None:
        params = {}
    if combine:
        return CombinedClassifier(lambda: Classifier(clf_model, params))
    return Classifier(clf_model, params)