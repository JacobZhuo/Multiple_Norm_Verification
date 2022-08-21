# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:46:18 2021

@author: win10
"""

import torch
import numpy as np
import pickle
from scipy import stats
from scipy.special import ndtr

def get_dm_TEP(m_type,pcs=16):
    import TEdataset
    if 'dnn' in m_type:
        dataloader, data = TEdataset.get_dataloader(seed=3691,ind=range(22))
        if 'ibp' in m_type:
            net = torch.load('TEP_models/best_acc_ibp.pkl')
        elif 'pgd' in m_type:
            net = torch.load('TEP_models/best_acc_pgd.pkl')
        elif 'distill' in m_type:
            net = torch.load('TEP_models/student.pkl')
        else:
            net = torch.load('TEP_models/best_acc32.pkl')

        model = next(net.children())
        prediction = net(torch.Tensor(data['x_test'])).max(1)[1]
        correct = torch.eq(prediction, torch.Tensor(data['y_test'])).numpy()
        clean_data = zip(data['x_test'][correct],data['y_test'][correct].astype(np.int32))
        print('Clean Accuracy%.4f'%np.mean(correct))
        print('************************')
        return model,clean_data
    if m_type=='svm':
        dataloader, data = TEdataset.get_dataloader(seed=3691,ind=range(22))
        f=open('TEP_models/svm.pkl','rb').read()
        clf = pickle.loads(f)
        # clf.score(data['x_train'],data['y_train'])
        w=clf.coef_
        b=clf.intercept_
        prediction = clf.predict(data['x_test'])
        correct = prediction == data['y_test']
        clean_data = zip(data['x_test'][correct],data['y_test'][correct].astype(np.int32))
        model=(w,b)
        print('Clean Accuracy%.4f'%np.mean(correct))
        print('************************')
        return model,clean_data
    if m_type=='pca':
        dataloader, data = TEdataset.get_dataloader_dect(seed=3691,ind=range(22),stand='nor')
        data_mean = np.mean(data['x_train'],0)
        data_std = np.std(data['x_train'],0)
        data_nor = (data['x_train'] - data_mean)/data_std
        data_test = (data['x_test'] - data_mean)/data_std
        X = np.cov(data_nor.T)
        P,v,P_t = np.linalg.svd(X)
        num_pc =pcs
        P = P[:,:num_pc]
        O1 = np.sum((v[:num_pc])**1)
        O2 = np.sum((v[:num_pc])**2)
        O3 = np.sum((v[:num_pc])**3)
        h0 = 1 - (2*O1*O3)/(3*(O2**2))
        # c_95 = 1.645
        c_99 = 2.325
        # SPE_95_limit = O1*((h0*c_95*((2*O2)**0.5)/O1 + 1 + O2*h0*(h0-1)/(O1**2))**(1/h0))
        SPE_99_limit = O1*((h0*c_99*((2*O2)**0.5)/O1 + 1 + O2*h0*(h0-1)/(O1**2))**(1/h0))
        I = np.eye(50)
        # delta = (I -np.dot(P, P.T)).T @ (I -np.dot(P, P.T))
        delta = (I -np.dot(P, P.T))
        q_total = []
        for x in data_test:
            q = x @ delta @ x.T
            # np.sum((x @ delta)**2)
            q_total.append(q)
        correct = ~np.bitwise_xor((q_total > SPE_99_limit),data['y_test'].astype(np.bool))
        # correct.mean()
        correct[data['y_test']!=0].mean()
        correct[data['y_test']==0].mean()
        clean_data = zip(data['x_test'][correct],data['y_test'][correct].astype(np.int32))
        model = (delta, SPE_99_limit, data_mean, data_std)
        print('Clean Accuracy%.4f'%np.mean(correct))
        print('************************')
        return model,clean_data
    if m_type=='ae':
        def cal_q(data,net):
            x_hat = net(torch.tensor(data).float())
            sqe=(torch.tensor(data)-x_hat)**2
            sqe=sqe.detach().numpy()
            q = np.sum(sqe,axis = 1)
            return q

        def cal_limit(q_normal):
            mean_q = np.mean(q_normal)
            std_q = np.std(q_normal)**2
            freedom = (2*(mean_q**2))/std_q
            chi_lim = stats.chi2.ppf(0.99,freedom)
            q_limit = std_q/(2*mean_q)*chi_lim
            return q_limit

        dataloader, data = TEdataset.get_dataloader_dect(seed=3691,ind=range(22),stand=True)
        net = torch.load('TEP_models/best_fc.pkl')
        model = next(net.children())
        q_norm=cal_q(data['x_train'],net)
        q_limit=cal_limit(q_norm)
        q_total = cal_q(data['x_test'],net)
        correct = ~np.bitwise_xor((q_total > q_limit),data['y_test'].astype(np.bool))
        clean_data = zip(data['x_test'][correct],data['y_test'][correct].astype(np.int32))
        model = (model, q_limit)
        print('Clean Accuracy%.4f'%np.mean(correct))
        print('************************')
        return model,clean_data
def get_dm_SP(m_type,pcs=2):
    import SPdataset
    if 'dnn' in m_type:
        dataloader,data=SPdataset.get_dataloader(2882)
        if 'ibp' in m_type:
            net = torch.load('SP_models/best_acc_ibp.pkl')
        elif 'distill' in m_type:
            net = torch.load('SP_models/student.pkl')
        elif 'pgd' in m_type:
            net = torch.load('SP_models/best_acc_pgd.pkl')
        else:
            net = torch.load('SP_models/best_acc_nor.pkl')
        model = next(net.children())
        prediction = net(torch.Tensor(data['x_test'])).max(1)[1]
        correct = torch.eq(prediction, torch.Tensor(data['y_test'])).numpy()
        clean_data = zip(data['x_test'][correct],data['y_test'][correct].astype(np.int32))
        print('Clean Accuracy%.4f'%np.mean(correct))
        print('************************')
        return model,clean_data
    if m_type=='svm':
        dataloader,data=SPdataset.get_dataloader(2882)
        f=open('SP_models/svm.pkl','rb').read()
        clf = pickle.loads(f)
        # clf.score(data['x_train'],data['y_train'])
        w=clf.coef_
        b=clf.intercept_
        prediction = clf.predict(data['x_test'])
        correct = prediction == data['y_test']
        clean_data = zip(data['x_test'][correct],data['y_test'][correct].astype(np.int32))
        model=(w,b)
        print('Clean Accuracy%.4f'%np.mean(correct))
        print('************************')
        return model,clean_data
    if m_type=='pca':
        dataloader,data=SPdataset.get_dataloader_dect(2882,stand='nor')
        data_mean = np.mean(data['x_train'],0)
        data_std = np.std(data['x_train'],0)
        data_nor = (data['x_train'] - data_mean)/data_std
        data_test = (data['x_test'] - data_mean)/data_std
        X = np.cov(data_nor.T)
        P,v,P_t = np.linalg.svd(X)
        num_pc = pcs
        P = P[:,:num_pc]
        O1 = np.sum((v[:num_pc])**1)
        O2 = np.sum((v[:num_pc])**2)
        O3 = np.sum((v[:num_pc])**3)
        h0 = 1 - (2*O1*O3)/(3*(O2**2))
        c_95 = 1.645
        # c_99 = 2.325
        SPE_95_limit = O1*((h0*c_95*((2*O2)**0.5)/O1 + 1 + O2*h0*(h0-1)/(O1**2))**(1/h0))
        # SPE_99_limit = O1*((h0*c_99*((2*O2)**0.5)/O1 + 1 + O2*h0*(h0-1)/(O1**2))**(1/h0))
        I = np.eye(27)
        # delta = (I -np.dot(P, P.T)).T @ (I -np.dot(P, P.T))
        delta = (I -np.dot(P, P.T))
        q_total = []
        for x in data_test:
            q = x @ delta @ x.T
            # np.sum((x @ delta)**2)
            q_total.append(q)
        correct = ~np.bitwise_xor((q_total > SPE_95_limit),data['y_test'].astype(np.bool))
        # correct.mean()
        correct[data['y_test']!=0].mean()
        correct[data['y_test']==0].mean()
        clean_data = zip(data['x_test'][correct],data['y_test'][correct].astype(np.int32))
        model = (delta, SPE_95_limit, data_mean, data_std)
        print('Clean Accuracy%.4f'%np.mean(correct))
        print('************************')
        return model,clean_data
    if m_type=='ae':
        def cal_q(data,net):
            x_hat = net(torch.tensor(data).float())
            sqe=(torch.tensor(data)-x_hat)**2
            sqe=sqe.detach().numpy()
            q = np.sum(sqe,axis = 1)
            return q

        def cal_limit(q_normal):
            mean_q = np.mean(q_normal)
            std_q = np.std(q_normal)**2
            freedom = (2*(mean_q**2))/std_q
            chi_lim = stats.chi2.ppf(0.99,freedom)
            q_limit = std_q/(2*mean_q)*chi_lim
            return q_limit

        dataloader,data=SPdataset.get_dataloader_dect(2882,stand=True)
        net = torch.load('SP_models/best_fc.pkl')
        model = next(net.children())
        q_norm=cal_q(data['x_train'],net)
        q_limit=cal_limit(q_norm)
        q_total = cal_q(data['x_test'],net)
        correct = ~np.bitwise_xor((q_total > q_limit),data['y_test'].astype(np.bool))
        clean_data = zip(data['x_test'][correct],data['y_test'][correct].astype(np.int32))
        model = (model, q_limit)
        print('Clean Accuracy%.4f'%np.mean(correct))
        print('************************')
        return model,clean_data