import numpy as np
import random
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split, cross_validate
import matplotlib.pyplot as plt

#########################################################
################## LINEAR REGRESSION ####################
#########################################################


def perform_linreg():

    # import data
    data = np.loadtxt('HW2Dataset.csv', delimiter=',', skiprows=1)
    X = np.hstack((np.ones((len(data), 1)), data[:, 1:16]))
    y = data[:, 16]

    # Split the data into a training and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    N_train = np.shape(X_train)[0]
    N_test = np.shape(X_test)[0]

    # print(np.shape(X_train))
    # print(np.linalg.matrix_rank(X_train))

    # Create and fit each of the three models to the training data
    model_linreg = LinearRegression(fit_intercept=False)
    model_linreg.fit(X_train, y_train)
    model_ridge = Ridge(alpha=.01, fit_intercept=False)
    model_ridge.fit(X_train, y_train)
    model_lasso = Lasso(alpha=.01)
    model_lasso.fit(X_train, y_train)

    # Print the model coefficients
    print("Model Coefficients: ",  model_linreg.coef_)
    # print(model_ridge.coef_)
    # print(model_lasso.coef_)
    # what = np.linalg.inv(X_train.T@X_train)@X_train.T@y_train


    plt.plot(X_test[:, 3], model_linreg.predict(X_test), 'bo')
    plt.plot(X_test[:, 3], y_test, 'go')
    plt.show()

    # Perform a k-fold cross validation and return the R2 values from each. Default is 5 folds
    cv_results_linreg = cross_validate(model_linreg, X_train, y_train)
    print("Linear Regression 5-fold R2 values:", cv_results_linreg['test_score'])
    cv_results_ridge = cross_validate(model_ridge, X_train, y_train)
    print("Ridge 5-fold R2 values:", cv_results_ridge['test_score'])
    cv_results_lasso = cross_validate(model_lasso, X_train, y_train)
    print("Lasso 5-fold R2 values:", cv_results_lasso['test_score'])

    # Calculate MSE between model and data for traininga and test sets
    MSE_train = np.sum((model_linreg.predict(X_train)-y_train)**2)/N_train
    print("Mean Square Error Regression Train: ", MSE_train)
    MSE_test = np.sum((model_linreg.predict(X_test)-y_test)**2)/N_test
    print("Mean Square Error Regression Test: ", MSE_test)

    MSE_train_ridge = np.sum((model_ridge.predict(X_train)-y_train)**2)/N_train
    print("Mean Square Error Ridge Train: ", MSE_train_ridge)
    MSE_test_ridge = np.sum((model_ridge.predict(X_test)-y_test)**2)/N_test
    print("Mean Square Error Ridge Test: ", MSE_test_ridge)

    MSE_train_lasso = np.sum((model_lasso.predict(X_train)-y_train)**2)/N_train
    print("Mean Square Error Lasso Train: ", MSE_train_lasso)
    MSE_test_lasso = np.sum((model_lasso.predict(X_test)-y_test)**2)/N_test
    print("Mean Square Error Lasso Test: ", MSE_test_lasso)

    print('-------------------------------------------------------------')

    return

#########################################################
################### GAUSSIAN PROCESS ####################
#########################################################


def perform_gaussian():

    # import data
    data = np.loadtxt('HW2Dataset.csv', delimiter=',', skiprows=1)
    X = data[:, 1:16]
    y = data[:, 16]

    # Split the data into a training and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)
    N_train = np.shape(X_train)[0]
    N_test = np.shape(X_test)[0]

    model_gaussian = GaussianProcessRegressor(random_state=1)
    model_gaussian.fit(X_train, y_train)
    print(model_gaussian.predict(X_train))

    # plt.plot(X_test[:, 3], model_gaussian.predict(X_test), 'bo')
    # plt.plot(X_test[:, 3], y_test, 'go')
    # plt.show()

    # Calculate MSE between model and data for traininga and test sets
    MSE_train = np.sum((model_gaussian.predict(X_train) - y_train) ** 2) / N_train
    print("Mean Square Error Regression Train: ", MSE_train)
    MSE_test = np.sum((model_gaussian.predict(X_test) - y_test) ** 2) / N_test
    print("Mean Square Error Regression Test: ", MSE_test)

    print('-------------------------------------------------------------')

    return


if __name__ == '__main__':

    # perform_linreg()
    perform_gaussian()
    # perform_nn()


