from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


class MyPerceptron():
    def __init__(self, n_iter, fit_intercept=True):
        """
        LinearRegression class.
 
        Attributes
        --------------------
            fit_intercept  --  Whether the intercept should be estimated or not.
            n_iter         --  Maximum number of iterations (in case of non-converging)
            coef_          --  The learned coeffient of the linear model
        """
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter
        self.coef_ = None

    def generate_features(self, X):
        """
        Returns pre-processed input data
        """
        if self.fit_intercept:
            ones = np.ones((len(X), 1))
            return np.concatenate((X, ones), axis=1)
        return X
    
    def fit(self, X, y):
        """
        Finds the coefficients of a linear model that fits the target.
 
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
 
        Returns
        --------------------
            self    -- an instance of self
        """
        X_ = self.generate_features(X)
        n, d = X_.shape

        # ********************************
        # implementation start from here --------------------------
        #  ********************************
        self.coef_ = np.zeros(d, dtype=np.float64)
        y[y == 0] = -1
        success = False

        curr_iter = 0
        while curr_iter < self.n_iter:
            check_vals = y * X_.dot(self.coef_)
            incorrect = np.where(check_vals <= 0)[0]
            if len(incorrect) != 0:
                i = incorrect[0]
                self.coef_ += y[i] * X_[i]
                curr_iter += 1
            else:
                print("Success after {} iterations!".format(curr_iter))
                success = True
                break

        if not success: print("Failed after {} iterations.".format(self.n_iter))
        # implementation end from here ----------------------------
        # ********************************

    def predict(self, X):
        """
        Predict output for X.
 
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
 
        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None:
            raise Exception("fit function not implemented")

        X_ = self.generate_features(X)
        y = np.dot(X_, self.coef_) >= 0
        return y


iris = datasets.load_iris()
X = iris.data[:100]
y = iris.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# A build-in perceptron model from scikit-learn
ppn = Perceptron(max_iter=40)
ppn.fit(X_train, y_train)
y_pred = ppn.predict(X_test)
print('Test Error: %.2f' % (1-accuracy_score(y_test, y_pred)))

# MyPerceptron
myppn = MyPerceptron(n_iter=40)
myppn.fit(X_train, y_train)
y_pred = myppn.predict(X_test)
print('Test Error: %.2f' % (1-accuracy_score(y_test, y_pred)))
