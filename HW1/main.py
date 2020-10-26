# https://towardsdatascience.com/machine-learning-basics-descision-tree-from-scratch-part-ii-dee664d46831

import csv
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
from matplotlib import cm

dataReader = csv.reader(open('synthetic-4.csv'), quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
numBins = 50

def countTarget(featureList):
    # counters[0] = rows, counters[1] = numZero, counters[2] = numOne
    counters = [0, 0, 0]
    for row in featureList:
        if row[2] == 0:
            counters[1] += 1
        elif row[2] == 1:
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

        for bin in np.unique(getCol(node.data)[3]):
            bin = int(bin)
            newData = []
            for row in node.data:
                if row[node.splitAttr] == bin:
                    newData.append(row)
            if node.splitAttr == -1:
                # No info gain
                node.isLeaf = True
                if node.count[1] > node.count[2]:
                    node.leafPrediction = 0
                else:
                    node.leafPrediction = 1
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
    prediction = -1
    for col in range(len(row) - 1):
        for child in node.children:
            if not child == None:
                if not child.isLeaf:
                    predict(child, row)
                else:
                    if row[col] == child.splitVal:
                        prediction = child.leafPrediction
                    if child.leafPrediction == row[-1] and row[col] == child.splitVal:
                        return 1, prediction
                    
    return 0, prediction

def accuracy(root, data):
    numCorrect = 0
    predictions = []
    for row in data:
        correct, prediction = predict(root, row)
        predictions.append(prediction)
        numCorrect += correct

    return numCorrect / len(data), predictions

def printTree(root):
    for child in root.children:
        if not child == None:
            if child.isLeaf:
                print("LEAF: ", child.splitVal, " | ", child.depth, " | ", child.leafPrediction)
                
            else:
                print("CHILD: ", child.splitVal, " | ", child.depth)
                printTree(child)

def graph(data, predictions):
    features = getCol(data)[3]
    targets = getCol(data)[2]

    plotStep = 0.02
    plotColors = "bry"

    for pairidx, pair in enumerate(features):
        if pairidx >= 6:
            continue
        X = features
        y = np.reshape(targets, len(targets))

        idx = np.arange(X.shape[0])
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        plt.subplot(2,3, pairidx + 1)

        xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
        ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(xmin, xmax, plotStep),
                            np.arange(ymin, ymax, plotStep))
        Z = [[el] for el in predictions]
        Z = np.hstack((data, Z))
        # cs = plt.contourf(xx, yy, Z, cmap=plt.get_cmap("Paired"))

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.axis("tight")

        for i, color in zip(range(numBins), plotColors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label="Target", cmap=plt.get_cmap("Paired"))
        plt.axis("tight")
    plt.suptitle("Synthetic-4 Decision Surface")
    plt.legend()
    plt.show()

def main():
    parent = []

    # Get parent
    for row in dataReader:
        parent.append(row)
        row[0] = int(row[0])
        row[1] = int(row[1])
        row[2] = int(row[2])
    targets = getCol(parent)[2]
    data = discretize(parent)
    rowIt = 0
    for row in data:
        row[len(row) - 1] = int(targets[rowIt])
        rowIt += 1
    root = Node(data, 0)
    ID3(root)
    print(accuracy(root, root.data)[0])

if __name__ == "__main__":
    main()