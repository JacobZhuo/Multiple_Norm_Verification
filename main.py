#python -m pip install -i https://pypi.gurobi.com gurobipy
import torch
from intervalbound import IntervalFastLinBound, IntervalBound
from milp import MILPVerifier_dnn,MILPVerifier_svm,MILPVerifier_pca,MILPVerifier_ae,MILPVerifier_ae_vy
import cvxpy as cp
import numpy as np
import sys
import gurobipy
import model_data
import dccp
import time
# print(gurobipy.__file__)
import matplotlib.pyplot as plt

def calc_radius(clean_data,m_radius,norm,m_type, model,fix0=None):
    # in_shape = 50
    if 'dnn' in m_type:
        prebound = IntervalFastLinBound(model, in_shape, 0, 1)
        bound = MILPVerifier_dnn(model, in_shape, 0, 1)
    if m_type in ['ae']:
        prebound = IntervalFastLinBound(model[0], in_shape, 0, 1)
        bound = MILPVerifier_ae(model, in_shape, in_min, in_max)
    if m_type in ['svm']:
        bound = MILPVerifier_svm(model, in_shape, 0, 1)
    if m_type in ['pca']:
        bound = MILPVerifier_pca(model, in_shape, 0, 1)

    history = dict()
    history['m_type'] = m_type
    history['norm'] = norm
    history['fix0'] = fix0
    history['m_radius'] = m_radius
    history['datas'] = []

    for i,(x,y) in enumerate(clean_data):

        data = dict()
        data['x'] = x
        data['y'] = y
        values = []
        advs = []

        if m_type in ['dnn','ae','dnnibp','dnndistill']:
            prebound.calculate_bound(x, m_radius)
        print('data:#%d'%i)

        if m_type in ['dnn','ae','dnnibp','dnndistill']:
            bound.construct(prebound.l, prebound.u, x, m_radius, norm, fix0 ,provar = pv)
            bound.prepare_verify(y)
        else:
            bound.construct(x, m_radius, norm, fix0)
            bound.prepare_verify(y)
        try:
            if m_type in ['ae','pca'] and y==0:
                result = bound.prob.solve(method='dccp',solver=cp.GUROBI, verbose=False, TimeLimit=240, Threads=12)
            else:
                bound.prob.solve(solver=cp.GUROBI, verbose=False, TimeLimit=240, Threads=12)

        except:
            continue
        if bound.prob.status not in ['optimal','Converged','optimal_inaccurate']:
            values.append([fix0 if fix0 != 0 else 50,bound.prob.status])
            print(bound.prob.status)

        else:
            if m_type in ['ae','pca'] and y==0:
                res = result[0]
            else:
                res = bound.prob.value
            values.append([fix0 if fix0 != 0 else 50,res])
            pert = bound.cx.value-x
            pert[abs(pert)<1e-8]=0
            advs.append(pert)
            print(res)
        print(bound.prob.status)

    data['values'] = values
    data['advs'] = advs
    history['datas'].append(data)
    return history

#%%
in_min, in_max = 0, 1
m_radius = 0.5
norm = -1 # Support 0, 1, 2, -1 (infty)
varepsion = None # None for single objective optimization, Integer for l_0-l_p optimization
m_type = 'svm' # Support 'svm', 'dnn', 'dnnibp', 'dnndistill', 'ae', 'pca'

model, clean_data = model_data.get_dm_SP(m_type)
# model, clean_data = model_data.get_dm_TEP(m_type)
in_shape = 27 # 50 for TEP

h = calc_radius(clean_data, m_radius, norm, m_type, model, fix0=varepsion)
