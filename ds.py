import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.colors import ListedColormap

#basic Perceptron for linear division of data
class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, resolution=0.01):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')


#dataf should have column teacher_column, linear division
def divide_frame(dataf,columns,teacher_column, random_state_sample = 1,random_state_ppn = 1,fraction = 0.7,eta = 0.1, iter = 50, plot = False,res = 0.1):
    dataf_extract = dataf.loc[:,columns+[teacher_column]]
    dataf_extract_train = dataf_extract.sample(random_state = random_state_sample, frac = fraction)
    dataf_extract_test = dataf_extract.drop(dataf_extract_train.index)
    
    X = dataf_extract_train.loc[:,columns].values
    X_test = dataf_extract_test.loc[:,columns].values
    y = dataf_extract_train.loc[:,teacher_column].values
    y_test = dataf_extract_test.loc[:,teacher_column].values
    pnp = Perceptron(eta,iter,random_state_ppn)

    pnp.fit(X,y)
    if plot:
        plot_decision_regions(X_test, y_test, classifier=pnp,resolution = res)
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plt.legend(loc='upper left')

    dataf_extract_test["prediction"] = pnp.predict(dataf_extract_test.loc[:,columns])
    errors = abs(dataf_extract_test["prediction"]-dataf_extract_test[teacher_column])/2
    
    percent = 100*errors.sum()/errors.count()
    print(errors.count(), f" total in test, errors: {percent:.2f}%")
    return pnp