#!/usr/bin/env python
import os
from random import shuffle
import pandas as pd
import numpy as np

def get_celebA_data(load_data_size=None):
    """Load the celebA dataset.
    Source: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    src_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(src_path, '../data/celebA/list_attr_celeba.csv'), sep=';')
    df = df.rename(columns={'Male': 'sex'})

    s = -1*df['sex']
    y = df['Smiling']
    df = df.drop(columns=['sex', 'Smiling','picture_ID'])

    X = df.to_numpy()
    y = y.to_numpy()
    s = s.to_numpy()

    if load_data_size is not None: # Don't shuffle if all data is requested
    	# shuffle the data
    	perm = list(range(0, len(y)))
    	shuffle(perm)
    	X = X[perm]
    	y = y[perm]
    	s = s[perm]

    	print("Loading only %d examples from the data" % load_data_size)
    	X = X[:load_data_size]
    	y = y[:load_data_size]
    	s = s[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    #remove duplicates
    _, unique_indices = np.unique(X, axis=0, return_index=True)

    return X[unique_indices,:], y[unique_indices], s[unique_indices]

def get_adult_data(load_data_size=None):
    """Load the Adult dataset.
    Source: UCI Machine Learning Repository.

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    def mapping(tuple):
        # native country
        tuple['native-country'] = "US" if tuple['native-country'] == "United-States" else "NonUS"
        # education
        if tuple['education'] in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
            tuple['education'] = "prim-middle-school"
        if tuple['education'] in ["9th", "10th", "11th", "12th"]:
            tuple['education'] = "high-school"
        return tuple


    src_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(src_path, '../data/adult/adult.csv'))
    df = df.drop(['race' ], axis=1)
    df = df.replace("?", np.nan)
    df = df.dropna()
    df = df.apply(mapping, axis=1)

    sensitive_attr_map = {'Male': 1, 'Female': -1}
    label_map = {'>50K': 1, '<=50K': -1}

    attrs = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
    int_attrs = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt']
    
    s = df['sex'].map(sensitive_attr_map).astype(int)
    y = df['income'].map(label_map).astype(int)


    x = pd.DataFrame(data=None)
    for x_var in attrs:
        x = pd.concat([x, pd.get_dummies(df[x_var], prefix=x_var, drop_first=False)], axis=1)
    for x_var in int_attrs:
        x = pd.concat([x, normalize(x=df[x_var])], axis=1)

    X = x.to_numpy()
    s = s.to_numpy()
    y = y.to_numpy()

    if load_data_size is not None: # Don't shuffle if all data is requested
        # shuffle the data
        perm = list(range(0, len(y)))
        shuffle(perm)
        X = X[perm]
        y = y[perm]
        s = s[perm]

        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        s = s[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    #remove duplicates
    _, unique_indices = np.unique(X, axis=0, return_index=True)

    return X[unique_indices,:], y[unique_indices], s[unique_indices]

def normalize(x):
	# scale to [-1, 1]
	x_ = (x - x.min()) / (x.max() - x.min()) * 2 - 1
	return x_

if __name__=='__main__':

    x,y,s = get_celebA_data()

    print(x.shape)
    print(s)
    print(y)
