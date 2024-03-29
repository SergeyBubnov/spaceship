import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#%pip install pydotplus
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from IPython.display import Image, display
from sklearn.preprocessing import StandardScaler


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


class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

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
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

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
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,  
            # in the case of logistic regression (as we will see later), 
            # we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

class LogisticRegressionGD(object):
    """Logistic Regression Classifier using gradient descent.

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
    cost_ : list
      Logistic cost function value in each epoch.

    """
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

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
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            
            # note that we compute the logistic `cost` now
            # instead of the sum of squared errors cost
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # equivalent to:
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

def plot_decision_regions(X, y, classifier, resolution=0.01,show = 0):
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    if(show):
      X = X[y == show]
      y = y[y == show]

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
def divide_frame(dataf,columns,teacher_column, classifier = 'Perceptron', kernel_ = 'rbf', #linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
                  random_state_sample = 1,random_state_ppn = 1, fraction = 0.7,
                  eta = 0.1, iter = 50,
                  gamma_ = 1, c = 1,
                  degree_ = 3,
                  neighbors = 5, dimension  = 2, metric_='minkowski',
                  criterion_tree = "gini", depth = 5,
                  plot = False,res = 0.1, show_value = 0):
    
    dataf_extract = dataf.loc[:,columns+[teacher_column]]
    if fraction<1:
      dataf_extract_train = dataf_extract.sample(random_state = random_state_sample, frac = fraction)
      dataf_extract_test = dataf_extract.drop(dataf_extract_train.index)
    else:
      dataf_extract_train = dataf_extract_test = dataf_extract

    X = dataf_extract_train.loc[:,columns].values
    X_test = dataf_extract_test.loc[:,columns].values
    y = dataf_extract_train.loc[:,teacher_column].values
    y_test = dataf_extract_test.loc[:,teacher_column].values
    if classifier == 'Perceptron':
      pnp = Perceptron(eta,iter,random_state_ppn)
    elif classifier == 'Adaline':
      pnp = AdalineGD(eta,iter,random_state_ppn)
    elif classifier == "LogisticRegression":
      pnp = LogisticRegressionGD(eta,iter,random_state_ppn)
    elif classifier == "SVC":
      pnp = SVC(kernel= kernel_, random_state=random_state_ppn, gamma=gamma_, C=c, degree = degree_)
    elif classifier == "KNN":
      pnp = KNeighborsClassifier(n_neighbors=neighbors, p=dimension, metric=metric_)
    elif classifier == "Tree":
      pnp = DecisionTreeClassifier(criterion=criterion_tree, max_depth=depth, random_state=random_state_ppn)
    else:
      print('Unexpected type of classifier')

    pnp.fit(X,y)
    if plot:
        plot_decision_regions(X_test, y_test, classifier=pnp,resolution = res,show= show_value)
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plt.legend(loc='upper left')
    
    if(classifier == 'Tree'):
      dot_data = export_graphviz(pnp, filled = True, rounded = True, class_names = ['-1.0', '1.0'], feature_names = columns, out_file = None)
      graph = graph_from_dot_data(dot_data)
      im = Image(graph.create_jpeg())
      display(im)

    dataf_extract_test["prediction"] = pnp.predict(dataf_extract_test.loc[:,columns])
    errors = (dataf_extract_test["prediction"]-dataf_extract_test[teacher_column])/2
    
    table = pd.DataFrame(index = [-1,1], columns = [-1,1])
    
    table.loc[-1,1] = 100*abs((errors<0).sum())/errors.count()
    table.loc[1,-1] = 100*abs((errors>0).sum())/errors.count()
    
    arr = dataf_extract_test["prediction"]
    
    for i in arr.index:
      if errors[i]!=0:
        arr[i] = 0
    
    table.loc[-1,-1] = 100*abs((arr < 0).sum())/errors.count()
    table.loc[1,1] = 100*abs((arr > 0).sum())/errors.count()
    
    errors = abs(errors)
    percent = 100*errors.sum()/errors.count()
    print("test size: ", errors.count(), f", total errors in test: {percent:.2f}%")
    print("index = predicted, columns = factual, in %:")
    
    print(table)

    if (classifier == 'Perceptron' or classifier == 'Adaline' or classifier == 'LogisticRegression'): 
      print("     w = ",pnp.w_)

    return pnp

def LDA_eigen(X,y,plot):

  if plot:
    np.set_printoptions(precision = 4)
    print("unique values of y: ",np.unique(y),"\n")
  
  mean_vecs = []
  labels = np.unique(y)
  for label in range(1,len(labels)+1):
      mean_vecs.append(np.mean(X[y == labels[label-1]],axis = 0))
      if plot:
        print('MV %s: %s\n'%(label,mean_vecs[label-1]))

  d = len(X[0])
  S_W = np.zeros((d,d))
  for label , mv in zip(labels , mean_vecs):
      class_scatter = np.zeros((d,d))
      for row in X[y == label]:
          row, mv = row.reshape(d,1), mv.reshape(d,1)
          class_scatter += (row - mv).dot((row - mv).T)
      S_W += class_scatter
  if plot:
    print ('Матрица раcсеяния внутри классов: %sx%s'%(S_W.shape[0], S_W.shape[1]))

  S_W = np.zeros((d,d))
  for label,mv in zip(labels, mean_vecs):
      class_scatter = np.cov(X[y==label].T)
      S_W += class_scatter
  if plot:
    print('Масштабированная матрица рассеяния внутри классов: %sx%s' % (S_W.shape[0], S_W.shape[1]) )

  mean_overall = np.mean(X, axis= 0)
  S_B = np.zeros((d,d))
  for i , mean_vec in enumerate(mean_vecs):
      n = X[y==i+1,:].shape[0]
      mean_vec = mean_vec.reshape(d,1)
      mean_overall = mean_overall.reshape(d,1)
      S_B +=n*(mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
  if plot:
    print('Матрица рассеяния между классами: %sx%s'%(S_B.shape[0],S_B.shape[1]))

  eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

  eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
  eigen_pairs=sorted(eigen_pairs, key = lambda k: k[0], reverse = True)

  if plot:
    print('Собственные значения в порядке убывания:\n')
    for eigen_val in eigen_pairs:
        print(eigen_val[0])

  tot = sum(eigen_vals.real)
  discr = [(i/tot) for i in sorted (eigen_vals.real, reverse = True)]
  cum_discr = np.cumsum(discr)
  size = d+1
  if plot:
    plt.bar(range(1,size),discr, alpha = 0.5, align = 'center',label = 'индивидуальная "Различимость"')
    plt.ylabel("Коэффициент 'различимости'")
    plt.xlabel('Линейные дискриминанты')
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.show()
  return eigen_pairs



class LDA:

  def __init__(self, plot = False):

    self.plot = plot
    self.eigen_pairs = []
    self.w = []

  def fit(self,X,y = None,params = None):
    #print("LDA fit\n")
    self.eigen_pairs = LDA_eigen(X,y,self.plot)
    self.w = self.eigen_pairs[0][1][:,np.newaxis].real
    if self.plot:
      print('Матрица W:\n',self.w)
        
    return self
    
  def transform(self,X,params = None):
    #print("LDA transform\n")
    return X.dot(self.w)

  def fit_transform(self,X,y = None,params = None):
    #print("LDA fit_transform\n")
    return self.fit(X,y).transform(X)
