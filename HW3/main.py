import numpy as np

examples = np.genfromtxt('data/mnist_train_0_1.csv', delimiter=',')
test = np.genfromtxt('data/mnist_test_0_1.csv', delimiter=',')
key = examples[:, 0]
X = np.delete(examples, 0, 1)
testKey = test[:, 0]
testX = np.delete(test, 0, 1)
rng = np.random.default_rng()
numH = 5
alpha = 0.1
numIt = 1

w1 = 2 * rng.random((len(X[0]) + 1, numH)) - 1
w2 = 2 * rng.random((numH + 1, 1)) - 1

# TRAINING
def normalize(x):
    return x / 255

for it in range(numIt):
    currRow = 0
    first = True
    for row in X:
        # Bias values
        xb = rng.choice([-1, 1])
        hb = rng.choice([-1, 1])
        row = list(map(normalize, row))
        row.append(xb)
        h = (1 / (1 + np.exp(0 - (row @ w1))))
        h = np.append(h, hb)
        o = (1 / (1 + np.exp(0 - (h @ w2))))[0]
        
        # Delta values
        deltaO = (key[currRow] - o) * (o * (1 - o))
        deltas = []
        deltas.append(deltaO)
        for currH in range(len(h) - 1):
            delta = (h[currH] * (1 - h[currH]) * deltaO * w2[currH])[0]
            deltas.append(delta)
            # Backprop
            w2[currH][0] = w2[currH][0] + (alpha * deltaO * h[currH])
            for wx in range(len(row) - 1):
                newWeight = w1[:, currH][wx] + (alpha * delta * row[wx])
                w1[:, currH][wx] = newWeight
                
        currRow += 1

        if first:
            print(row @ w1)
            print(deltas)
            first = False
            
            
    # TESTING
    currRow = 0
    numCorrect = 0
    num0 = 0
    num1 = 0
    first = True
    for row in testX:
        # Bias values
        xb = rng.choice([-1, 1])
        hb = rng.choice([-1, 1])
        row = list(map(normalize, row))
        row.append(xb)
        h = (1 / (1 + np.exp(0 - (row @ w1))))
        h = np.append(h, hb)
        o = (1 / (1 + np.exp(0 - (h @ w2))))[0]
        if first:
            print(w2)
            first = False
        if o <= 0.5:
            o = 0
            num0 += 1
        else:
            o = 1
            num1 += 1
        if o == testKey[currRow]:
            numCorrect += 1
        currRow += 1
    accuracy = numCorrect / currRow
    percentage = "{:.0%}".format(accuracy)
    print("\nEPOCH: ", str(it))
    print("Neural Net's accuracy was ", percentage)
    print("Number of 0s: ", str(num0), " | Number of 1s: ", str(num1), " | Out of: ", currRow)