"""
Apply a SearchFair classifier to the UCI Credit dataset

Authors: Robin Guillot, Grégoire Retourné, Rémi Turquier
"""

import pandas as pd
from searchfair import SearchFair
from sklearn.model_selection import train_test_split

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
kernel = 'linear'
verbose = True
reg_beta = 0.0001

linear_model_DDP = SearchFair(reg_beta=reg_beta,
                              kernel=kernel,
                              fairness_notion=fairness_notion,
                              verbose=verbose,
                              stop_criterion=0.01)
linear_model_DDP.fit(x_train, y_train, s_train=s_train)
