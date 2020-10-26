import numpy as np
import matplotlib.pyplot as plt
import csv

def yhat(X, theta):
    return X @ theta

def loss(X, y, theta):
    return np.mean(np.square(yhat(X, theta) - y))

def main():
    reader = csv.reader(open('data/synthetic-3.csv'), quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
    origX = []
    origy = []
    examples = []
    
    for row in reader:
        origX.append(row[0])
        origy.append(row[1])
        examples.append(row)
    
    alpha = 0.001
    numIt = 1000
    order = 7

    X = np.array([origX]).T
    y = np.array(origy)
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    # Add polynomial features   
    for i in range(2,order+1):
        X = np.hstack((X, (X[:, 1] ** i).reshape((m, 1))))
        n += 1

    # Normalize features
    X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)

    # Create theta
    rng = np.random.default_rng()
    theta = rng.random(n+1)
    theta = theta - alpha * (1/m)

    # Train with gradient descent
    losses = []
    for i in range(numIt):
        theta = theta - alpha * (1/m) * (X.T @ ((X @ theta) - y))
        losses.append(loss(X, y, theta))

    # Graph results
    hx = []
    linespace = np.linspace(min(X[:, 1]), max(X[:, 1]), len(X[:, 1]))
    plt.scatter(linespace, origy)
    for point in linespace:
        currh = 0
        for i in range(len(theta)):
            if i == 0:
                currh += theta[0]
            else:
                currh += theta[i]*point**i
        hx.append(currh)

    
    plt.plot(linespace, hx)
    weightIt = 0
    for weight in theta:
        if weight == theta[0]:
            suptitle = str(round(theta[0], 2))
        else:
            suptitle += (" + " + str(round(weight, 2)) + "x^" + str(weightIt))
        weightIt += 1
    
    plt.suptitle(suptitle, fontsize=10)
    plt.title("Mean Squared Error: " + str(round((sum(losses) / len(losses)), 2)))
    plt.show()


if __name__ == "__main__":
    main()