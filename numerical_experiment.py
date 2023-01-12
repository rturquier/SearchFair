"""
Apply a SearchFair classifier to the UCI Credit dataset

Authors: Robin Guillot, Grégoire Retourné, Rémi Turquier
"""

import pandas as pd
from searchfair import SearchFair

data_url = ("https://archive.ics.uci.edu/ml/"
            + "machine-learning-databases/credit-screening/crx.data")
column_renamer = {index: "A" + str(index + 1) for index in range(16)}

credit_df = (
    pd.read_csv(data_url, header=None)
    .rename(columns=column_renamer)
    .loc[:, ["A1", "A9", "A10", "A12", "A16"]]
    .replace(regex={"\?": None, "[at\+]": True, "[bf\-]": False})
    .convert_dtypes()
)

y_data = credit_df.A1
x_data = credit_df.loc[:, ["A9", "A10", "A12"]]
s_data = credit_df.A16