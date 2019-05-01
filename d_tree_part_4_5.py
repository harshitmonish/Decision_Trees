# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 20:30:44 2018

@author: harshitm
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import read_data as rd

def get_data(data):
    y = data[:,0]
    x = data[:,1:]
    return x, y

def find_accuracy(result, y):
    N = len(y)
    sum_ = 0
    for i in range(0,N):
        if y[i] == result[i]:
            sum_ += 1
    return sum_/N

def random_forest(train, valid, test):
    x_train, y_train = get_data(train)
    x_test, y_test = get_data(test)
    x_valid, y_valid = get_data(valid)
    rf = RandomForestClassifier(n_estimators=200,max_features=9 , bootstrap=True)
    #rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    valid_pred = rf.predict(x_valid)
    test_pred = rf.predict(x_test)
    
    train_acc = find_accuracy(train_pred, y_train)
    valid_acc = find_accuracy(valid_pred, y_valid)
    test_acc = find_accuracy(test_pred, y_test)
    
    print("Train accuracy is : ", train_acc)
    print("Valid accuracy is : ", valid_acc)
    print("Test accuracy is : ", test_acc)

def d_trees(train, valid, test):
    x_train, y_train = get_data(train)
    x_test, y_test = get_data(test)
    x_valid, y_valid = get_data(valid)
    
    #clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=8, min_samples_leaf=5, min_samples_split=2)
    #clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth=10,min_samples_leaf=5, min_samples_split=2)
    clf_entropy = DecisionTreeClassifier(criterion = "entropy")
    clf_entropy.fit(x_train, y_train)
    y_pred_en_test = clf_entropy.predict(x_test)
    y_pred_en_train = clf_entropy.predict(x_train)
    y_pred_en_valid = clf_entropy.predict(x_valid)
    test_accuracy = find_accuracy(y_pred_en_test, y_test)   
    train_accuracy = find_accuracy(y_pred_en_train, y_train)
    valid_accuracy = find_accuracy(y_pred_en_valid, y_valid)
    print("Train accuracy is : ", train_accuracy)
    print("Valid accuracy is : ", valid_accuracy)
    print("Test accuracy is : ", test_accuracy)
    
def main():
 
    train_data = rd.train_data
    test_data = rd.test_data
    valid_data = rd.valid_data
    
    #d_trees(train_data, test_data, valid_data)
    random_forest(train_data, valid_data, test_data)

if __name__ == "__main__":
    main()