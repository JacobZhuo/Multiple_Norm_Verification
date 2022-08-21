import numpy as np
import torch
from sklearn import preprocessing
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import  Counter
import numpy.ma as ma
rootdir = "TE/"
batch_size=32

def get_dataloader(seed,ind):
    data,labs=get_xy(ind)
    data = preprocessing.MinMaxScaler().fit_transform(data)

    x_train, x_test, y_train, y_test = train_test_split(data, labs, test_size=0.30, random_state=seed)

    nparray=dict()
    nparray['x_train']=x_train
    nparray['x_test']=x_test[::10,:]
    nparray['y_train']=y_train
    nparray['y_test']=y_test[::10]

    x_train=torch.from_numpy(x_train).float()
    x_test=torch.from_numpy(x_test).float()
    y_train=torch.LongTensor(y_train)
    y_test=torch.LongTensor(y_test)

    train_dataset=torch.utils.data.TensorDataset(x_train,y_train)
    test_dataset=torch.utils.data.TensorDataset(x_test[::10,:],y_test[::10])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
    data_loader = dict()
    data_loader['train'] = train_loader
    data_loader['test'] = test_loader
    return data_loader,nparray

def get_dataloader_dect(seed,ind, stand = False):
    data,labs=get_xy(ind)
    if stand:
        data = preprocessing.StandardScaler().fit_transform(data)
        data = preprocessing.MinMaxScaler().fit_transform(data)
    if stand == 'nor':
        data = preprocessing.MinMaxScaler().fit_transform(data)
    x_train, x_test, y_train, y_test = train_test_split(data, labs, test_size=0.30, random_state=seed)

    nparray=dict()
    tind = np.arange(0,x_test.shape[0],10)
    remain = np.delete(x_test,tind,0)
    remain_y = np.delete(y_test,tind,0)
    remain_nor = remain[remain_y==0,:]
    remain_nor = remain_nor[:30,:]

    nparray['x_train']=x_train[y_train==0]
    nparray['x_test']=np.vstack((x_test[::10,:],remain_nor))
    nparray['y_train']=y_train[y_train==0]
    nparray['y_test']=np.hstack((y_test[::10],np.zeros(remain_nor.shape[0])))
    nparray['y_test'][nparray['y_test']!=0]=1

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

def get_fault(ind):
    data=sio.loadmat(rootdir+'TE_wch_50_IDV_{:d}.mat'.format(ind))
    fl=data['TE_wch_IDV_{:d}'.format(ind)]
    fl=fl[::10,:]
    fl=np.delete(fl,[0,46,50,53],axis=1)
    return fl

def get_xy(ind):
    X=np.empty(shape=(0, 50))
    y=[]
    for n,i in enumerate(ind):
        fl=get_fault(i)
        X=np.append(X,fl,axis=0)
        y=np.append(y,(n)*np.ones(fl.shape[0]))
    return X,y

