import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, DotProduct
from sklearn.model_selection import train_test_split, cross_validate
import tensorflow as tf
from tensorflow import keras
from keras import layers
# print(tf.__version__)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    N_train = np.shape(X_train)[0]
    N_test = np.shape(X_test)[0]

    # f = plt.figure()
    # ax = f.add_subplot()
    # ax.plot(X_train[:, 1], y_train, 'bo', markersize=1)
    # ax.set_xlabel('Factor 2')
    # ax.set_ylabel('Response')
    # plt.show()


    # Create the gaussian model and fit
    length_scale = np.ones(15)
    kernel = RBF(length_scale=length_scale) + WhiteKernel()
    model_gaussian = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, random_state=5, normalize_y=True)
    model_gaussian.fit(X_train, y_train)
    print(f"Hyperparameters: {model_gaussian.kernel_}")
    # print(model_gaussian.predict(X_train))

    # idx = np.argsort(X_test[:, 0])
    # plt.plot(X_test[:, 0][idx], model_gaussian.predict(X_test[idx]), 'b--')
    # plt.plot(X_test[:, 0], y_test, 'go')
    # plt.show()

    # Calculate MSE between model and data for traininga and test sets
    MSE_train = np.sum((model_gaussian.predict(X_train) - y_train) ** 2) / N_train
    print("Mean Square Error Regression Train: ", MSE_train)
    MSE_test = np.sum((model_gaussian.predict(X_test) - y_test) ** 2) / N_test
    print("Mean Square Error Regression Test: ", MSE_test)

    print('-------------------------------------------------------------')

    return

#########################################################
################### NEURAL NETWORK ######################
#########################################################


def perform_nn():

    # import data
    raw_data = pd.read_csv('HW2Dataset.csv')
    dataset = raw_data.copy().drop(columns=raw_data.columns[0], axis=1)

    # print(dataset.head())


    # split into training and test sets
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    print(train_dataset.describe().transpose())

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('stiffness_value')
    test_labels = test_features.pop('stiffness_value')

    # normalize the data
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    PC1 = np.array([train_features['PC1']])
    PC1_normalizer = layers.Normalization(input_shape=[1,], axis=None)
    PC1_normalizer.adapt(PC1)
    test_results = {}

    def plot_loss(history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0, 40])
        plt.xlabel('Epoch')
        plt.ylabel('Error [stiffness_value]')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_PC1(x, y):
        plt.scatter(train_features['PC1'], train_labels, label='Data', s=1)
        plt.plot(x, y, color='k', label='Predictions')
        plt.xlabel('PC1')
        plt.ylabel('stiffness_value')
        plt.legend()
        plt.show()

    def build_and_compile_model(norm):
        model = keras.Sequential([
            norm,
            layers.Dense(64, activation='relu'),
            layers.Dense(64 , activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.0001))
        return model

    dnn_PC1_model = build_and_compile_model(PC1_normalizer)
    dnn_PC1_model.summary()
    history = dnn_PC1_model.fit(
        train_features['PC1'],
        train_labels,
        validation_split=0.2,
        verbose=0,
        epochs=40
    )
    plot_loss(history)
    x = tf.linspace(-25, 35, 61)
    y = dnn_PC1_model.predict(x)
    plot_PC1(x, y)

    test_results['dnn_PC1_model'] = dnn_PC1_model.evaluate(test_features['PC1'], test_labels, verbose=0)

    dnn_model = build_and_compile_model(normalizer)
    dnn_model.summary()

    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        verbose=0,
        epochs=100
    )

    plot_loss(history)
    test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
    print(pd.DataFrame(test_results, index=['Mean absolute error [stiffness_value]']).T)

    test_prediction = dnn_model.predict(test_features).flatten()
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_prediction, s=2)
    plt.xlabel('True Values [stiffness_value]')
    plt.ylabel('Predictions [stiffness_value]')
    lims = [0, 110]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    error = test_prediction - test_labels
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [stiffness_value]')
    _ = plt.ylabel('Count')
    plt.show()


    print('-------------------------------------------------------------')

    return

if __name__ == '__main__':

    # perform_linreg()
    # perform_gaussian()
    perform_nn()

