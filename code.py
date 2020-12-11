# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:19:09 2020

@author: ggpen
"""
import numpy as np

from collections import Counter

import pandas as pd

from sklearn.tree import DecisionTreeClassifier




class Random:

    def __init__(self, n_trees=100, max_depth=100, min_sample_split=2, n_feats=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.min_sample_split = min_sample_split
        self.trees = []

    def boot_strap(self, x, y):
        n = x.shape[0]
        ind = np.random.choice(n, n, replace=True)
        return x[ind], y[ind]

    def fit(self, x, y):
        for i in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_sample_split)
            x_b, y_b = self.boot_strap(x, y)
            tree.fit(x_b, y_b)
            self.trees.append(tree)

    def predict(self, x):
        preds = np.array([tree.predict(x) for tree in self.trees])
        preds = np.swapaxes(preds, 0, 1)
        return np.array([Counter(pred).most_common(1)[0][0] for pred in preds])

    def score(self, x, y):
        pred = self.predict(x)
        return 1 - np.sum(np.abs((pred - y))) / len(y)


class KNN:

    def __init__(self, k=3):
        self.k = k

    def distance(self, x, x2):
        return np.sqrt(np.sum((x - x2) ** 2))

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        return np.array([self.help_pred(i) for i in x])

    def help_pred(self, x):
        distances = [self.distance(x, x2) for x2 in self.x_train]
        k_ind = np.argsort(distances)[:self.k]
        knn = [self.y_train[k] for k in k_ind]
        return Counter(knn).most_common(1)[0][0]

    def score(self, x, y):
        pred = self.predict(x)
        return 1 - np.sum(np.abs((pred - y))) / len(y)


class NaiveBayes:

    def fit(self, x, y):
        n_rows, n_columns = x.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.mean = np.zeros((n_rows, n_classes), dtype=np.float64)
        self.var = np.zeros((n_rows, n_classes), dtype=np.float64)
        self.prior = np.zeros(n_classes, dtype=np.float64)
        for ind, c in enumerate(self.classes):
            x_c = x[c == y]
            self.mean[ind, :] = x_c.mean(axis=0)
            self.var[ind, :] = x_c.var(axis=0)
            self.prior[ind] = x_c.shape[0] / float(n_rows)

    def predict(self, x):
        return np.array([self.help_pred(i) for i in x])

    def help_pred(self, x):
        posts = []
        for ind, c in enumerate(self.classes):
            prior = np.log(self.prior[ind])
            post = np.sum(np.log(self.pdf(ind, x)))
            post += prior
            posts.append(post)
        return self.classes[np.argmax(posts)]

    def pdf(self, ind, x):
        mean = self.mean[ind]
        var = self.var[ind]
        p = np.exp(-(x - mean) ** 2 / (2 * var))
        q = np.sqrt(2 * np.pi * var)
        return p / q

    def score(self, x, y):
        pred = self.predict(x)
        return 1 - np.sum(np.abs((pred - y))) / len(y)


class AdaBoost:

    def fit(self, x, y, iters=100):
        n = x.shape[0]
        self.sample_weights = np.zeros((iters, n))
        self.stamps = np.zeros(iters, dtype=object)
        self.stamp_weights = np.zeros(iters)
        self.errors = np.zeros(iters)
        self.sample_weights[0] = np.ones(n) / n
        for t in range(iters):
            current_sample_weight = self.sample_weights[t]
            stamp = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            stamp = stamp.fit(x, y)
            stamp_pred = stamp.predict(x)
            err = current_sample_weight[stamp_pred != y].sum()
            stamp_weight = np.log((1 - err) / err) / 2
            new_sample_weight = current_sample_weight * np.exp(-stamp_weight * y * stamp_pred)
            new_sample_weight /= new_sample_weight.sum()
            if t + 1 < iters:
                self.sample_weights[t + 1] = new_sample_weight
            self.errors[t] = err
            self.stamps[t] = stamp
            self.stamp_weights[t] = stamp_weight

    def predict(self, x):
        preds = np.array([stamp.predict(x) for stamp in self.stamps])
        return np.sign(self.stamp_weights @ preds)

    def score(self, x, y):
        pred = self.predict(x)
        return 1 - np.sum(np.abs(pred - y)) / len(y)


class LogisticRegression1:

    def model(self, x, t):
        return x @ t

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient(self, y_pred, y, x):
        return ((y_pred - y) @ x) / len(y)

    def fit(self, x, y, lr=0.1, epochs=2000):
        t = np.random.random(x.shape[1])
        for epoch in range(epochs):
            h_x = self.model(x, t)
            y_pred = self.sigmoid(h_x)
            grad = self.gradient(y_pred, y, x)
            t = t - lr * grad
        self.coef_ = t

    def predict(self, x, trashold=0.5):
        y_pred = self.sigmoid((x @ self.coef_))
        pred = np.zeros(x.shape[0])
        pred[y_pred >= trashold] = 1
        return pred

    def score(self, x, y):
        y_pred = self.predict(x)
        return 1 - np.sum(np.abs(y_pred - y)) / len(y)


class Rando:

    def __init__(self, n_trees=100, max_depth=100, min_sample_split=2, n_feats=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_feats = n_feats
        self.trees = []

    def boot(self, x, y):
        n = x.shape[0]
        ind = np.random.choice(n, n, replace=True)
        return x[ind], y[ind]

    def fit(self, x, y):
        for i in range(self.n_trees):
            tree = DecisionTreeClassifier(min_samples_split=self.min_sample_split, max_depth=self.max_depth)
            x_b, y_b = self.boot(x, y)
            tree.fit(x_b, y_b)
            self.trees.append(tree)

    def predict(self, x):
        preds = np.array([tree.predict(x) for tree in self.trees])
        preds = np.swapaxes(preds, 0, 1)
        return np.array([Counter(pred).most_common(1)[0][0] for pred in preds])

    def score(self, x, y):
        y_pred = self.predict(x)
        return 1 - np.sum(np.abs(y_pred - y)) / len(y)


class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):

        approx = np.dot(X, self.w) - self.b
        pred = np.sign(approx)
        pred = np.where(pred == -1, 0, 1)
        return pred

    def score(self, x, y):
        pred = self.predict(x)
        return 1 - np.sum(np.abs((pred - y))) / len(y)

# this is updated