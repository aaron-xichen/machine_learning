#!/usr/bin/env python
# encoding: utf-8

from sklearn.decomposition import PCA
import os
import cPickle as pickle
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_data():
    path = os.path.expanduser("~/documents/dataset/MNIST/mnist.pkl.gz")
    print path
    f = gzip.open(path, 'rb')
    return pickle.load(f)[0]


def showImage(data):
    assert(data.shape[0] == 10)
    width = np.sqrt(data.shape[1])
    assert(data.shape[1] == width**2)
    image = [arr.reshape((width, width)) for arr in data]
    fig = plt.figure()
    for i in xrange(4):
        for j in xrange(3):
            idx = i*3+j
            if idx >= 10:
                continue
            ax = fig.add_subplot(4,3, idx+1)
            ax.imshow(image[idx], cmap=cm.Greys_r)
    fig.show()


def scale(data):
    min_value = np.min(data)
    max_value = np.max(data)
    return (data - min_value) / (max_value-min_value) *255


# use pca to reconstruct the input data, (n_examples, n_features)
# also will return the mean of each digits
def reconstruct(trin_x, n_components):
    pca = PCA(n_components)
    pca.fit(train_x)
    trans = np.transpose(pca.components_)
    reconstruct_x = reduce(np.dot, (train_x, trans, np.transpose(trans)))
    reconstruct_df = pd.DataFrame(scale(reconstruct_x))
    reconstruct_df = pd.concat([reconstruct_df, train_y_DF], axis=1)
    reconstruct_mean_df = reconstruct_df.groupby('label').apply(lambda x:np.mean(x, axis=0))
    loss_mean_sqrt_root(train_x, reconstruct_x)
    return reconstruct_mean_df

## cal the loss
def loss_mean_sqrt_root(train_x, reconstruct_x):
    loss = (train_x - reconstruct_x)**2
    loss_df = pd.DataFrame(loss)
    loss_df = pd.concat([loss_df, train_y_DF], axis=1)

    # cal the sqrt root mean of each digits
    loss_sqrt_root_mean_df = loss_df.groupby('label').apply(lambda x:np.mean(np.sqrt(np.mean(x, axis=1)),axis=0))
    # print loss_sqrt_root_mean_df
    print np.mean(loss_sqrt_root_mean_df)

# prepare training data
train_set = load_data()
train_x = train_set[0]
train_y = train_set[1]
trainDF = pd.DataFrame(scale(train_x))
train_y_DF = pd.DataFrame(data={'label':train_y})
trainDF = pd.concat([trainDF, train_y_DF], axis=1)

# cal the mean matrix of each digits
train_mean_DF = trainDF.groupby('label').apply(lambda x:np.mean(x, axis=0))

width = 5
height = 7
fig= plt.figure()
steps = [5,10,20,40,80, 160]
#plot origin data
for j in range(width):
    ax = fig.add_subplot(height, width, j+1)
    ax.set_axis_off()
    digit_origin= train_mean_DF.values[j,:-1]
    digit_origin= digit_origin.reshape((28,28))
    ax.imshow(digit_origin, cmap=cm.Greys_r,interpolation='nearest')

# plot the reconstruct image
for i, n_components in zip(range(1,len(steps)+1), steps):
    print 'process n_components={}'.format(n_components)
    reconstruct_mean_df = reconstruct(train_x, n_components)
    for j in range(width):
        ax = fig.add_subplot(height, width, i*width+j+1)
        ax.set_axis_off()
        digit_reconstruct = reconstruct_mean_df.values[j,:-1]
        digit_reconstruct = digit_reconstruct.reshape((28,28))
        ax.imshow(digit_reconstruct, cmap=cm.Greys_r, interpolation='nearest')

fig.show()
