# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 20:23:31 2018

@author: harshitm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import nltk 
#import re
import math
import copy
#from random import choice
#import time
import _pickle as cpickle
import os
import read_data as rd
import queue

class treenode():
    def __init__(self, is_leaf, children, pos, neg, parent, attrib, value, id):
        self.is_leaf = False
        self.children = {}
        self.pos = None
        self.neg = None
        self.parent = parent
        self.attrib = None
        self.value = None
        self.id = None

def calc_entropy(y):
    res = 0.0
    freq = {}
    len_y = len(y)
    for i in y:
        if(i in freq.keys()):
            freq[i] += 1.0
        else:
            freq[i] = 1.0
    for p in freq.values():
        res += -((float(p)/len_y) * (math.log((float(p)/len_y), 2)))
    return res
    

def info_gain(data, attr):
    y = data[:,0]
    #print(y,attr)
    entropy = calc_entropy(y)
    subentropy = 0.0
    freq = {}
    for row in data:
        if(row[attr] in freq.keys()):
            freq[row[attr]] += 1.0
        else:
            freq[row[attr]] = 1.0
    
    for val in freq.keys():
        y_vec = []
        prob_val = freq[val] / len(y)
        for record in data:
            if(record[attr] == val):
                y_vec.append(record[0])  
        subentropy += (prob_val * calc_entropy(y_vec))
    gain = entropy - subentropy
    return gain

def attr_choose(x, attributes, y):

    attrib_index = [i for i in range(len(attributes))]
    best = attrib_index[1]
    newGain = {}
    maxGain = 0;
    for attr in range(1, len(attributes)):
        gain = info_gain(x, attr)
        #print(gain)
        newGain[attr] = gain
        if gain > maxGain:
            maxGain = gain
            newGain[attr] = gain
            #best = attr

    for attr, gain in newGain.items():
        if gain == maxGain:
            best = attr
            break
    return attributes[best]

def get_values(x, attributes, attr):

    index = attributes.index(attr)
    values = []

    for entry in x:
        if entry[index] not in values:
            values.append(entry[index])

    return values

def get_data(x, attributes, best, val):

    new_data = [[]]
    index = attributes.index(best)

    for entry in x:
        if (entry[index] == val):
            newEntry = []
            for i in range(0,len(entry)):
                if(i != index):
                    newEntry.append(entry[i])
            new_data.append(newEntry)

    new_data.remove([])   
    return new_data
count = 0
count_arr = []
accuracy_train_arr = []
accuracy_test_arr = []
accuracy_valid_arr = []

root = treenode(True, None, None, None, None, None, None, 0)
attributes_glob = []
def build_tree(x, attributes, parent_node, y, train_data, valid_data, test_data, pick_saved):
    global acuuracy_train_arr
    global count
    global count_arr
    global root
    if(pick_saved == True):
        file_name = "objs/build_before_prune"
        if os.path.exists(file_name):
            fileObj = open(file_name, 'rb')
            root = cpickle.load(fileObj)
            return root
        else:
            print("Obj file not found, :( ")
            return
    if(len(attributes) - 1 == 0):
        node_dummy = treenode(True, None, None, None, parent_node, None, None, 0)
        node_dummy.parent.is_leaf = True
        node_dummy.is_leaf = True
        node_dummy.attrib = "Dummy"
        node_dummy.value = x[0]
        node_dummy.children = None
        #node_dummy.height = 0
        return node_dummy
    
    else:
        node = treenode(True, None, None, None, parent_node, None, None, 0)
        count+= 1

        #accuracy = predict()
        count_arr.append(count)
        #print("count is:", count)
        vals = [record[attributes.index(y)] for record in x]       
        pos_l = [i for i in vals if i == 1]
        neg_l = [i for i in vals if i == 0]
        node.pos = len(pos_l)
        node.neg = len(neg_l)
        if(parent_node is None):
            root = node
        #print(node.pos)
        if(node.pos == 0 or node.neg == 0):
            node.attrib = "Final_leaf"
            node.is_leaf = True
            node.children = None
            if(node.pos > node.neg):
                node.value = 1
            else:
                node.value = 0
            return node
        
        node.is_leaf = False
        best = attr_choose(x, attributes, y)
        node.attrib = best
        node_values = get_values(x, attributes, best)
        for val in node_values:
            new_data = get_data(x, attributes, best, val)
            new_data = np.array(new_data)
            newAttr = attributes[:]
            newAttr.remove(best)
            
            subnodes = build_tree(new_data, newAttr, node, y, train_data, valid_data, test_data, pick_saved)
            subnodes.id = val
            if(node.pos > node.neg):
                node.value = 1
            else:
                node.value = 0
            if subnodes.attrib != "Dummy":
                node.children[val] = subnodes
    file_name = "objs/build_before_prune" 
    fileObj = open(file_name,'wb')    
    arr = root     
    cpickle.dump(arr, fileObj)
    return node

def test_example(example, node, attributes_glob):
    if node.is_leaf == True:
        return node.value
    index = attributes_glob.index(node.attrib)
    test_val = example[index]
    if test_val not in node.children.keys():
        return node.value
    else:
        val = test_example(example, node.children[test_val], attributes_glob)
    return val
    
def predict(x, attributes, root):
    result = []
    for entry in x:
        root_temp = copy.copy(root)
        res = test_example(entry, root_temp, attributes)
        #print("result is : ", res)
        result.append(res)
        
    return result

def find_accuracy_and_predict(x, root):
    global attributes_glob
    result = []
    for entry in x:
        root_temp = copy.copy(root)
        res = test_example(entry, root_temp, attributes_glob)
        #print("result is : ", res)
        result.append(res)
    accuracy = find_accuracy(result, x[:,0])
    return accuracy

def calc_count_accuracy(x, root, pick_saved, val):
    if(pick_saved):
        if val == 1:
            file_name = "objs/train_acc"
        elif val == 2:
            file_name = "objs/valid_acc"
        elif val == 3:
            file_name = "objs/test_acc"
            
        if os.path.exists(file_name):
            fileObj = open(file_name, 'rb')
            arr = cpickle.load(fileObj)
            count = arr[0]
            accuracy_arr = arr[1]
            return count, accuracy_arr
        else:
            print("Obj file not found, :( ")
            return
        
    accuracy_arr = []
    counter = []
    new_root = treenode(True, None, None, None, None, None, None, 0)
    que = queue.Queue()
    #que.put(root)
    temp_node = root
    count = 1
    new_root.attrib = root.attrib
    new_root.id = root.id
    new_root.is_leaf = root.is_leaf
    new_root.neg = root.neg
    new_root.pos = root.pos
    new_root.parent = root.parent
    new_root.value = root.value
    temp_new_root = new_root
    counter.append(1)
    accuracy_arr.append(find_accuracy_and_predict(x, new_root))
    #print("it is : ",accuracy_arr[0])
    #print("here")
    while(1):
        count += 1
        if(temp_node.children):
            for key in temp_node.children.keys():
                temp_new_root.children[key] = temp_node.children[key]
                accur = find_accuracy_and_predict(x, new_root)
                #print(accur, count)
                accuracy_arr.append(accur)
                counter.append(count)
                que.put(temp_node.children[key])
        if(not que.empty()):
            temp_node = que.get()
            temp_new_root = temp_node
        if(count > 20):
            break
    print(len(counter))
    if val == 1:
        file_name = "objs/train_acc"
    elif val == 2:
        file_name = "objs/valid_acc"
    elif val == 3:
        file_name = "objs/test_acc" 
    fileObj = open(file_name,'wb')    
    arr = [counter, accuracy_arr]     
    cpickle.dump(arr, fileObj)
    return counter, accuracy_arr
   
#def modify_tree(root, l):
        
def get_height(root):
    if(root is None):
        return 0
    elif(not root.children):
        return 1
    else:
        depths = []
        for key in root.children.keys():
            depths.append(get_height(root.children[key]))
        return(max(depths) + 1)
attributes_glob = []     

def test_example(example, node, attributes_glob):
    if node.is_leaf == True:
        return node.value
    index = attributes_glob.index(node.attrib)
    test_val = example[index]
    if test_val not in node.children.keys():
        return node.value
    else:
        val = test_example(example, node.children[test_val], attributes_glob)
    return val
    
def predict(x, attributes, root):
    result = []
    for entry in x:
        root_temp = copy.copy(root)
        res = test_example(entry, root_temp, attributes)
        #print("result is : ", res)
        result.append(res)
        
    return result
    
def count_nodes(root):

    if root.children is not None:
        count = 1
        for key in root.children.keys():
            count += count_nodes(root.children[key])
        return count
    else:
        return 1  
    
def find_accuracy(result, y):
    N = len(y)
    sum_ = 0
    for i in range(0,N):
        if y[i] == result[i]:
            sum_ += 1
    return sum_/N

def validate_row(row, root, attributes):
    if(root.is_leaf == True or root.attrib == "Final_leaf"):
        predicted = root.value
        actual = row[0]
        if(predicted == actual):
            return 1
        else:
            return 0
        
    index = attributes.index(root.attrib)
    value = row[index]
    if value not in root.children.keys():
        if root.pos > root.neg:
            predicted = 1
        else:
            predicted = 0
        if(predicted == row[0]):
            return 1
        else:
            return 0
    #print("attrib, value", root.attrib, value)
    return validate_row(row, root.children[value], attributes)
        
def validate_tree(valid_data, root, attributes):
    len_valid_data = len(valid_data)
    valid = 0
    for entry in valid_data:
        valid += validate_row(entry, root, attributes)
    return valid/ len_valid_data

def prune_tree_final(root, node, valid_data, best_score, attributes, pick_saved):
    #print("here", best_score)
    if(pick_saved):
        file_name = "objs/build_after_prune"
        if os.path.exists(file_name):
            fileObj = open(file_name, 'rb')
            root = cpickle.load(fileObj)
            return root
        else:
            print("Obj file not found, :( ")
            return
    elif(node is None):
        return 
    elif(not node.children):
        return
    else:
        for key in node.children.keys():
            node.children[key].is_leaf = True
            new_score = validate_tree(valid_data, root, attributes)
            if new_score <= best_score:
                if node.children[key].attrib != "Final_leaf":
                    node.children[key].is_leaf = False
            else:
                node.children[key].children = None
                best_score = new_score
                print("new_score", new_score)
            prune_tree_final(root, node.children[key], valid_data, best_score, attributes, pick_saved)
    file_name = "objs/build_after_prune" 
    fileObj = open(file_name,'wb')    
    arr = root     
    cpickle.dump(arr, fileObj)
    return root

def plot_accuracy(a, b, file):
    plt.plot(a, b,color = "g")
    #plt.xlim(xmin = 0)
    plt.savefig(file) 
    plt.show()

def save_root(root):
    file_name = "objs/build_before_prune" 
    fileObj = open(file_name,'wb')    
    arr = root     
    cpickle.dump(arr, fileObj)

def adjust_data(a, b, count):
    print(count)
    for i in range(21, count):
        a.append(i)
        b.append(b[-1])
    return a, b
    
def main():
    train_data = rd.train_data
    test_data = rd.test_data
    valid_data = rd.valid_data
    headers = rd.headers
    target_attr = headers[0]
    global root
    root = build_tree(train_data, headers, root.parent, target_attr, train_data, valid_data, test_data, True)
    save_root(root)
    global attributes_glob
    attributes_glob = headers
    #print_tree(root)
    #compute_count_accuracy(train_data, headers, root)
    result = predict(test_data, headers, root)
    accuracy = find_accuracy(result, test_data[:,0])
    print("the accuracy is: ", accuracy)
    count = count_nodes(root)
    print("Nodes are : ", count)
    #plot_accuracy()
    best_score = validate_tree(valid_data, root, headers)
    print("best Score is :", best_score)
    new_root = copy.copy(root)
    #new_root_prune = prune_tree_final(new_root, new_root, valid_data, best_score, headers, True)
    #result_new = predict(test_data, headers, new_root_prune)
    #new_accuracy = find_accuracy(result_new, test_data[:,0])
    #print("new accurary : ", new_accuracy)
    
    #new_count = count_nodes(new_root_prune)
    #print("count is ", new_count)
    #counter_train, accuracy_train = calc_count_accuracy(train_data, new_root_prune, True, 1)
    #counter_valid, accuracy_valid = calc_count_accuracy(valid_data, new_root_prune, True, 2)
    #counter_test, accuracy_test = calc_count_accuracy(test_data, new_root_prune, True, 3)
    #counter_train_final, accuracy_train_final = adjust_data(counter_train, accuracy_train, count) 
    #counter_valid_final, accuracy_valid_final = adjust_data(counter_valid, accuracy_valid, count)
    #counter_test_final, accuracy_test_final = adjust_data(counter_test, accuracy_test, count) 
    #plot_accuracy(counter_train, accuracy_train, "train_graph_prune.png")
    #plot_accuracy(counter_valid, accuracy_valid, "valid_graph_prune.png")    
    #plot_accuracy(counter_test, accuracy_test, "test_graph_prune.png")
    
if __name__ == "__main__":
    main()