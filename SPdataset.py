# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 20:33:22 2020

@author: Jacob
"""


import numpy as np
import torch
from sklearn import preprocessing
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
batch_size=32

def get_dataloader(seed):
    txt=np.loadtxt('Faults.NNA')
    data=txt[:,:27]
    labs=np.argmax(txt[:,27:],axis=1)
    data = preprocessing.MinMaxScaler().fit_transform(data)

    x_train, x_test, y_train, y_test = train_test_split(data, labs, test_size=0.30, random_state=seed)

    nparray=dict()
    nparray['x_train']=x_train
    nparray['x_test']=x_test
    nparray['y_train']=y_train
    nparray['y_test']=y_test

    x_train=torch.from_numpy(x_train).float()
    x_test=torch.from_numpy(x_test).float()
    y_train=torch.LongTensor(y_train)
    y_test=torch.LongTensor(y_test)

    train_dataset=torch.utils.data.TensorDataset(x_train,y_train)
    test_dataset=torch.utils.data.TensorDataset(x_test,y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
    data_loader = dict()
    data_loader['train'] = train_loader
    data_loader['test'] = test_loader
    return data_loader,nparray

def get_dataloader_dect(seed = 2882,stand = True):
    txt=np.loadtxt('SPdataset.NNA')
    data=txt[:,:27]
    labs=np.argmax(txt[:,27:],axis=1)
    if stand:
        data = preprocessing.StandardScaler().fit_transform(data)
        data = preprocessing.MinMaxScaler().fit_transform(data)
    if stand == 'nor':
        data = preprocessing.MinMaxScaler().fit_transform(data)
    x_train, x_test, y_train, y_test = train_test_split(data[labs==0,:], labs[labs==0], test_size=0.30, random_state=seed)


    nparray=dict()
    nparray['x_train']=x_train
    td=data[labs!=0,:]
    nparray['x_test']=np.vstack((x_test,td[::5,:]))
    nparray['y_train']=y_train
    tl=labs[labs!=0]
    nparray['y_test']=np.hstack((y_test,np.ones_like(tl[::5])))

    x_train=torch.from_numpy(nparray['x_train']).float()
    x_test=torch.from_numpy(nparray['x_test']).float()
    y_train=torch.LongTensor(nparray['y_train'])
    y_test=torch.LongTensor(nparray['y_test'])

    train_dataset=torch.utils.data.TensorDataset(x_train,y_train)
    test_dataset=torch.utils.data.TensorDataset(x_test,y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
    data_loader = dict()
    data_loader['train'] = train_loader
    data_loader['test'] = test_loader
    return data_loader,nparray