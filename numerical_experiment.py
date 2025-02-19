"""
Apply a SearchFair classifier to the UCI Credit dataset

Authors: Robin Guillot, Grégoire Retourné, Rémi Turquier
"""

import pandas as pd
from searchfair import SearchFair
from sklearn.model_selection import train_test_split
import examples.utils as ut
import numpy as np
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

# %% ------ Functions -------
def print_clf_stats(model, x_train, x_test, y_train, y_test, s_train, s_test):
    """Print classifier stats (function written by Lohaus et al.)"""
    train_acc = ut.get_accuracy(np.sign(model.predict(x_train)), y_train)
    test_acc = ut.get_accuracy(np.sign(model.predict(x_test)), y_test)
    test_DDP, test_DEO = ut.compute_fairness_measures(model.predict(x_test),
                                                      y_test,
                                                      s_test)
    train_DDP, train_DEO = ut.compute_fairness_measures(model.predict(x_train),
                                                        y_train,
                                                        s_train)

    print(10*'-'+"Train"+10*'-')
    print("Accuracy: %0.4f%%" % (train_acc * 100))
    print("DDP: %0.4f%%" % (train_DDP * 100),
          "DEO: %0.4f%%" % (train_DEO * 100))
    print(10*'-'+"Test"+10*'-')
    print("Accuracy: %0.4f%%" % (test_acc * 100))
    print("DDP: %0.4f%%" % (test_DDP * 100),
          "DEO: %0.4f%%" % (test_DEO * 100))

# %% ------ Main code -------
data_url = ("https://archive.ics.uci.edu/ml/"
            + "machine-learning-databases/credit-screening/crx.data")
column_renamer = {index: "A" + str(index + 1) for index in range(16)}

credit_df = (
    pd.read_csv(data_url, header=None)
    .rename(columns=column_renamer)
    .loc[:, ["A1", "A9", "A10", "A12", "A16"]]
    .replace(regex={"\?": None, "[at\+]": 1, "[bf\-]": -1})
    .dropna()
)

y_data = credit_df.A1.to_numpy()
x_data = credit_df.loc[:, ["A9", "A10", "A12"]].to_numpy()
s_data = credit_df.A16.to_numpy()

x_train, x_test, y_train, y_test, s_train, s_test = (
    train_test_split(x_data, y_data, s_data,
                     train_size=0.80, shuffle=True)
)

fairness_notion = 'DDP'
kernel = 'rbf'
verbose = True
reg_beta = 0.0001

linear_model_DDP = SearchFair(reg_beta=reg_beta,
                              kernel=kernel,
                              fairness_notion=fairness_notion,
                              verbose=verbose,
                              stop_criterion=0.01)
linear_model_DDP.fit(x_train, y_train, s_train=s_train)

print_clf_stats(linear_model_DDP,
                x_train, x_test, y_train, y_test, s_train, s_test)


# %% Cross-validation
kernel = "rbf"
cv_model = SearchFair(kernel=kernel,
                      fairness_notion=fairness_notion,
                      verbose=0)

# regularization parameter beta
beta_params = [0.0001, 0.001, 0.01]
cv_params = {'reg_beta': beta_params}

if kernel == 'rbf':
    n_features = x_data.shape[1]
    default_width = 1/n_features
    order_of_magn = np.floor(np.log10(default_width))
    kernel_widths = [10**(order_of_magn), default_width, 10**(order_of_magn+1)]
    cv_params['gamma'] = kernel_widths

grid_clf = GridSearchCV(cv_model,
                        cv_params,
                        cv=3,
                        verbose=1,
                        n_jobs=1,
                        scoring='accuracy',
                        refit=True)
grid_clf.fit(x_train, y_train, s_train=s_train)

print(grid_clf.best_params_)

print_clf_stats(grid_clf,
                x_train, x_test, y_train, y_test, s_train, s_test)
