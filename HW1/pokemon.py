# https://towardsdatascience.com/machine-learning-basics-descision-tree-from-scratch-part-ii-dee664d46831

import csv
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
from matplotlib import cm

numBins = 8

def countTarget(featureList):
    # counters[0] = rows, counters[1] = numZero, counters[2] = numOne
    counters = [0, 0, 0]
    for row in featureList:
        if row[len(row) - 1] == 0:
            counters[1] += 1
        elif row[len(row) - 1] == 1:
            counters[2] += 1
    counters[0] = len(featureList)
    return counters

def allEntropy(data):
    counter = countTarget(data)
    ratio1 = counter[1] / counter[0]
    ratio2 = counter[2] / counter[0]
    if ratio1 == 0.0 or ratio2 == 0.0:
        return 0.0
    return 0 - (ratio1 * math.log(ratio1, 2) + ratio2 * math.log(ratio2, 2))

class Node:
    def __init__(self, data, depth):
        self.children = []
        self.data = data
        self.depth = depth
        self.splitVal = -1
        self.splitAttr = -1
        self.leafPrediction = -1
        self.count = countTarget(self.data)
        self.rows = self.count[0]
        if self.rows == 0:
            self.isLeaf = True
            if self.count[1] > self.count[2]:
                self.leafPrediction = 0
            else:
                self.leafPrediction = 1
        else:
            self.entropy = allEntropy(data)
            
            if self.entropy == 0 or depth >= 3:
                self.isLeaf = True
                
                if self.count[1] > self.count[2]:
                    self.leafPrediction = 0
                else:
                    self.leafPrediction = 1
            else:
                self.isLeaf = False
        

    def addChild(self, newNode):
        self.children.append(newNode)

def splitEntropy(data, splitAttr):
    # data = node.data
    parentEntropy = allEntropy(data)
    entropySum = 0
    for i in range(numBins):
        split = []
        for row in data:
            if row[splitAttr] == i:
                split.append(row)
        if split:
            entropySum += (allEntropy(split) * (len(split) / len(data)))
    return parentEntropy - entropySum

def testSplit(node):
    bestInfo = 0
    bestAttr = -1
    for col in range(len(node.data[0]) - 1):
        info = splitEntropy(node.data, col)
        if info > bestInfo:
            bestInfo = info
            bestAttr = col
    return bestAttr


def discretize(allRows):
    # Sort into bins using sklearn
    
    discretizer = KBinsDiscretizer(n_bins=numBins, encode='ordinal', strategy='uniform')
    discretizer.fit(allRows)
    newRows = discretizer.transform(allRows)
            
    return newRows

def ID3(node):
    if node.isLeaf:
        return
    elif node.rows == 0:
        return
    elif node.count[1] == 0 or node.count[2] == 0:
        return
    else:
        node.splitAttr = testSplit(node)

        for bin in range(numBins):
            bin = int(bin)
            newData = []
            for row in node.data:
                if row[node.splitAttr] == bin:
                    newData.append(row)
            if node.splitAttr == -1:
                # No info gain
                node.isLeaf = True
                if node.count[1] > node.count[2]:
                    node.leafPrediction = 'False'
                else:
                    node.leafPrediction = 'True'
                return

            newNode = Node(newData, node.depth + 1)
            newNode.splitVal = bin
            if not newNode.isLeaf:
                node.addChild(newNode)
                ID3(newNode)
            else:
                node.addChild(newNode)
    return

def getCol(data):
    feature1 = []
    feature2 = []
    allFeatures = []
    target = []
    for col in range(len(data[0])):
        for row in data:
            if col == 0:
                feature1.append(row[col])
            elif col == 1:
                feature2.append(row[col])
            else:
                target.append(int(row[col]))
    allFeatures = np.vstack((feature1, feature2)).T
    return feature1, feature2, target, allFeatures

def predict(node, row):

    for child in node.children:
        if not child == None:
            if not child.isLeaf:
                predict(child, row)
            else:
                if child.leafPrediction == row[-1]:
                    return 1
                    
    return 0

def accuracy(root, data):
    numCorrect = 0
    for row in data:
        numCorrect += predict(root, row)

    return numCorrect / len(data)

def printTree(root):
    for child in root.children:
        if not child == None:
            if child.isLeaf:
                print("LEAF: ", child.splitVal, " | ", child.depth, " | ", child.leafPrediction)
                
            else:
                print("CHILD: ", child.splitVal, " | ", child.depth)
                printTree(child)

def main():
    stats = []
    legendary = []

    # Get parent
    with open('pokemonStats.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            stats.append([int(i) for i in row])
    
    with open('pokemonLegendary.csv', 'r') as csvfile:
        legendaryReader = csv.reader(open('pokemonLegendary.csv'))
        next(legendaryReader)
        for row in legendaryReader:
            isLegendary = row[0]
            if isLegendary == "True":
                legendary.append([True])
            elif isLegendary == "False":
                legendary.append([False])

    stats = discretize(stats)
    data = np.hstack((stats, legendary))
    root = Node(data, 0)
    ID3(root)
    print(accuracy(root, data))

if __name__ == "__main__":
    main()